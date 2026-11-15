/**
 * RootCause Disease Classifier Service
 * Primary: Live FastAPI Backend Model Integration (MobileNetV2 / TensorFlow)
 * Offline: Disease guides remain available without a diagnosis.
 */

import { findDiseaseByPrediction } from '../data/diseases.js';
import { getApiBaseUrl as resolveApiBaseUrl } from '../config.js';

export class ClassifierService {
  /**
   * Determine the API base URL dynamically
   */
  static getApiBaseUrl() {
    return resolveApiBaseUrl();
  }

  /**
   * Check if backend API server is online and healthy
   * @returns {Promise<{online: boolean, status: string, url: string, latencyMs?: number}>}
   */
  static async checkBackendHealth() {
    const baseUrl = this.getApiBaseUrl();
    const startTime = Date.now();
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);
      let res;
      try {
        res = await fetch(`${baseUrl}/health`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          signal: controller.signal
        });
      } finally {
        clearTimeout(timeoutId);
      }

      if (res.ok) {
        const data = await res.json();
        const latencyMs = Date.now() - startTime;
        return {
          online: data.status === 'healthy',
          status: data.status || 'healthy',
          url: baseUrl,
          latencyMs
        };
      }
      return { online: false, status: 'error', url: baseUrl };
    } catch (err) {
      return { online: false, status: 'unreachable', url: baseUrl, error: err.message };
    }
  }

  /**
   * Convert image source (DataURL or local asset URL) to a Blob for multipart upload
   */
  static async imageSourceToBlob(imageSource) {
    if (!imageSource) return null;

    if (imageSource instanceof Blob) {
      return imageSource;
    }

    if (typeof imageSource === 'string' && imageSource.startsWith('data:')) {
      const comma = imageSource.indexOf(',');
      if (comma < 0) throw new Error('Selected image data is malformed');
      const header = imageSource.slice(0, comma);
      const payload = imageSource.slice(comma + 1);
      const mime = header.match(/^data:([^;,]+)/i)?.[1] || 'image/jpeg';
      const bstr = atob(payload);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      const blob = new Blob([u8arr], { type: mime });
      return /image\/svg\+xml/i.test(mime) ? await this.rasterizeImageBlob(blob) : blob;
    }

    // URL path (e.g. assets/images/sample_leaf_rice.svg)
    const response = await fetch(imageSource);
    if (!response.ok) throw new Error(`Could not load selected image (${response.status})`);
    const blob = await response.blob();
    const isSvg = /image\/svg\+xml/i.test(blob.type) || /\.svg(?:[?#]|$)/i.test(imageSource);
    return isSvg ? await this.rasterizeImageBlob(blob) : blob;
  }

  /** Convert vector sample images to a raster format supported by the API. */
  static async rasterizeImageBlob(blob, maxDimension = 800) {
    const objectUrl = URL.createObjectURL(blob);
    try {
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error('Selected image could not be decoded by this browser'));
        image.src = objectUrl;
      });

      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Image conversion is unavailable in this browser');
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      return await new Promise((resolve, reject) => {
        canvas.toBlob(
          result => result ? resolve(result) : reject(new Error('Could not convert the selected image')),
          'image/jpeg',
          0.9
        );
      });
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  }

  /**
   * Run disease detection on a captured leaf image
   * @param {Object} params
   * @param {string|Blob} params.imageData - Data URL, image URL, or Blob
   * @param {string} params.crop - 'rice' | 'wheat'
   * @returns {Promise<Object>} Diagnostic result
   */
  static async classifyLeaf({ imageData, crop = 'rice' }) {
    const baseUrl = this.getApiBaseUrl();
    const normalizedCrop = String(crop || '').trim().toLowerCase();
    if (!['rice', 'wheat'].includes(normalizedCrop)) {
      return { isUnavailable: true, crop: normalizedCrop, cropDisplayName: 'Unsupported crop', message: 'Choose Rice or Wheat to run a diagnosis.', disease: null };
    }

    // 2. PRIMARY PATH: Attempt live FastAPI prediction if backend is reachable
    let backendError = null;
    try {
      const imageBlob = await this.imageSourceToBlob(imageData);
      if (imageBlob) {
        const formData = new FormData();
        formData.append('crop_type', normalizedCrop);
        // Ensure filename extension is acceptable (.jpg/.png)
        const filename = imageBlob.type === 'image/png' ? 'leaf.png' : 'leaf.jpg';
        formData.append('image', imageBlob, filename);

        const controller = new AbortController();
        // First inference can include TensorFlow startup and selected-model load
        // on a CPU-only Windows host; allow a cold start before aborting.
        const timeoutId = setTimeout(() => controller.abort(), 90000);
        let response;
        try {
          response = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
          });
        } finally {
          clearTimeout(timeoutId);
        }

        const responseData = await response.json();

        // Case A: Image quality check failed
        if (response.status === 400 && responseData.status === 'quality_check_failed') {
          return {
            isBackend: true,
            isQualityFailed: true,
            crop: normalizedCrop,
            cropDisplayName: normalizedCrop === 'rice' ? 'Rice (धान)' : 'Wheat (गेहूं)',
            message: responseData.message || 'Image failed quality assessment.',
            issues: responseData.issues || [],
            suggestions: responseData.suggestions || [
              'Please hold the phone steady, tap to focus on the leaf, and retake the photo.'
            ],
            qualityAssessment: responseData.quality_assessment || null
          };
        }

        // Case B: Successful backend inference
        if (response.ok && responseData.status === 'success') {
          if (responseData.prediction?.status === 'success') {
            return this.formatBackendSuccessResult(responseData, normalizedCrop);
          }
          backendError = responseData.prediction?.message || 'The trained model could not complete this prediction.';
        }

        // Case C: Model not available on server
        if (responseData.status === 'model_not_available' || responseData.prediction?.status === 'model_not_available') {
          console.warn('Backend model not deployed for crop:', responseData.message);
          backendError = responseData.message || responseData.prediction?.message || 'The trained model is unavailable.';
        } else if (!response.ok || responseData.status !== 'success') {
          backendError = responseData.message || responseData.prediction?.message || `Prediction failed (${response.status})`;
        }
      }
    } catch (err) {
      console.warn('Backend prediction request failed:', err.message);
      backendError = err.message;
    }

    // Keep offline guides available, but never present a heuristic as a diagnosis.
    return {
      isBackend: false,
      isUnavailable: true,
      crop: normalizedCrop,
      cropDisplayName: normalizedCrop === 'rice' ? 'Rice (धान)' : 'Wheat (गेहूं)',
      message: backendError
        ? `A diagnosis could not be completed: ${backendError}`
        : 'The prediction service is unavailable. Reconnect to the configured RootCause server and try again.',
      disease: null
    };
  }

  /**
   * Format live FastAPI backend prediction into RootCause contract
   */
  static formatBackendSuccessResult(backendData, crop) {
    const prediction = backendData.prediction || {};
    const predictedDiseaseName = prediction.prediction;
    const confidenceVal = Number(prediction.confidence);
    if (!predictedDiseaseName || !Number.isFinite(confidenceVal) || confidenceVal < 0 || confidenceVal > 1) {
      return { isBackend: false, isUnavailable: true, crop, cropDisplayName: crop === 'rice' ? 'Rice (धान)' : 'Wheat (गेहूं)', message: 'The server did not return a valid prediction.', disease: null };
    }
    const confidencePct = parseFloat((confidenceVal * 100).toFixed(1));

    // Match to local catalog for multilingual text, treatment dosages, and icons
    let matchedDisease = findDiseaseByPrediction(crop, predictedDiseaseName);

    // If backend provided its own advisory, merge it
    const backendAdvisory = backendData.advisory;
    let symptoms = backendAdvisory?.symptoms || matchedDisease?.symptoms || [];
    let summary = backendAdvisory?.monitoring_advice || matchedDisease?.summary || '';
    let scientificName = backendAdvisory?.scientific_name || matchedDisease?.scientificName || '';
    let severity = backendAdvisory?.severity_level || matchedDisease?.severity || 'Moderate';

    const isHealthy = predictedDiseaseName && predictedDiseaseName.toLowerCase().includes('healthy');

    return {
      isBackend: true,
      isDemo: false,
      isOfflineFallback: false,
      isInconclusive: confidencePct < 55.0 && !isHealthy,
      crop: crop,
      cropDisplayName: crop === 'rice' ? 'Rice (धान)' : 'Wheat (गेहूं)',
      diseaseId: matchedDisease ? matchedDisease.id : 'backend_' + crop,
      diseaseName: predictedDiseaseName || (matchedDisease ? matchedDisease.name : 'Unknown Symptom'),
      scientificName: scientificName,
      confidence: confidencePct,
      severity: severity,
      status: isHealthy
        ? 'Healthy Plant - No Disease Detected'
        : (severity === 'Critical' ? 'Urgent Attention Required' : 'Confirmed Infection'),
      basicSymptoms: summary,
      disease: (matchedDisease ? {
        ...matchedDisease,
        symptoms,
        summary,
        scientificName,
        severity,
        treatment: {
          organic: backendAdvisory?.organic_controls || matchedDisease.treatment.organic,
          chemical: backendAdvisory?.chemical_guidance ? [backendAdvisory.chemical_guidance] : matchedDisease.treatment.chemical
        },
        prevention: backendAdvisory?.preventive_measures || matchedDisease.prevention,
        expertWarning: backendAdvisory?.disclaimer || matchedDisease.expertWarning
      } : null) || {
        id: 'custom_prediction',
        name: predictedDiseaseName,
        scientificName: scientificName,
        crop: crop,
        cropDisplayName: crop === 'rice' ? 'Rice (धान)' : 'Wheat (गेहूं)',
        severity: severity,
        season: 'Current Crop Cycle',
        priority: severity === 'Critical' ? 'Urgent / Critical' : 'Standard',
        summary: summary,
        symptoms: symptoms,
        visibleSigns: backendAdvisory?.monitoring_advice || 'Visible leaf discoloration.',
        causes: 'Pathogen detected by trained neural network model.',
        treatment: {
          organic: backendAdvisory?.organic_controls || ['Apply recommended bio-control agents.'],
          chemical: [backendAdvisory?.chemical_guidance || 'Consult local extension office for approved formulations.']
        },
        prevention: backendAdvisory?.preventive_measures || ['Follow balanced crop nutrition.'],
        expertWarning: backendAdvisory?.disclaimer || 'Consult your local Krishi Vigyan Kendra before chemical spray.',
        badgeClass: severity === 'Critical' ? 'badge-critical' : (isHealthy ? 'badge-healthy' : 'badge-high')
      },
      qualityAssessment: backendData.quality_assessment || null,
      rawScores: prediction.raw_scores || [],
      backendSource: 'Trained MobileNetV2 Deep Learning Model'
    };
  }

}

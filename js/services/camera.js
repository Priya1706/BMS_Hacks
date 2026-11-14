/**
 * RootCause Camera & Image Upload Service
 * Manages device camera stream, photo capture, and gallery image processing
 */

export class CameraService {
  static stream = null;

  /**
   * Start live camera stream
   * @param {HTMLVideoElement} videoElement
   * @returns {Promise<boolean>}
   */
  static async startCamera(videoElement) {
    try {
      this.stopCamera(); // Clean up existing stream

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('Camera API not supported on this device/browser.');
        return false;
      }

      const constraints = {
        video: {
          facingMode: { ideal: 'environment' }, // Prefer rear camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.stream = stream;
      if (videoElement) {
        videoElement.srcObject = stream;
        await videoElement.play();
      }
      return true;
    } catch (err) {
      console.warn('Could not access live camera:', err.name, err.message);
      return false;
    }
  }

  /**
   * Capture current frame from live video element
   * @param {HTMLVideoElement} videoElement
   * @returns {string|null} Data URL of captured image
   */
  static captureFrame(videoElement) {
    if (!videoElement || videoElement.videoWidth === 0) return null;

    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.85);
  }

  /**
   * Stop active camera stream
   */
  static stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  /**
   * Read and optimize uploaded image file from gallery
   * @param {File} file
   * @param {number} maxWidth
   * @param {number} maxHeight
   * @returns {Promise<string>} Data URL
   */
  static processImageFile(file, maxWidth = 800, maxHeight = 800) {
    return new Promise((resolve, reject) => {
      if (!file || !file.type.startsWith('image/')) {
        return reject(new Error('Selected file is not an image'));
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          // Scale down image to conserve storage and memory
          let width = img.width;
          let height = img.height;

          if (width > maxWidth || height > maxHeight) {
            if (width > height) {
              height = Math.round((height * maxWidth) / width);
              width = maxWidth;
            } else {
              width = Math.round((width * maxHeight) / height);
              height = maxHeight;
            }
          }

          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);

          const optimizedDataUrl = canvas.toDataURL('image/jpeg', 0.82);
          resolve(optimizedDataUrl);
        };
        img.onerror = () => reject(new Error('Failed to load image into memory'));
        img.src = e.target.result;
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsDataURL(file);
    });
  }
}

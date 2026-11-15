/**
 * RootCause - Main Application Controller
 * Handles screen navigation, camera integration, offline classification,
 * multilingual dictionary rendering, audio advisory, and local persistence.
 */

import { DISEASES_DATABASE, getAllDiseases, getDiseaseById, getDiseasesByCrop } from './data/diseases.js';
import { TRANSLATIONS, SUPPORTED_LANGUAGES, getTranslation } from './data/translations.js';
import { StorageService } from './services/storage.js';
import { ClassifierService } from './services/classifier.js';
import { AudioService } from './services/audio.js';
import { CameraService } from './services/camera.js';
import { assessFieldObservations, getQuestionnaireOptions } from './services/contextualAssessment.js';

class RootCauseApp {
  constructor() {
    this.currentScreen = 'screen-welcome';
    this.screenHistory = [];
    this.currentCrop = 'rice'; // 'rice' | 'wheat'
    this.currentImageData = null;
    this.currentDiagnosis = null;
    this.currentDisease = null;
    this.questionnaireSession = null;
    this.isViewingCatalogDisease = false;
    this.currentLanguage = 'en';
    this.userSettings = StorageService.getSettings();

    // Default sample images for quick evaluation
    this.SAMPLE_IMAGES = {
      rice: 'assets/images/sample_leaf_rice.svg',
      wheat: 'assets/images/sample_leaf_wheat.svg'
    };
  }

  /**
   * Initialize application components and listeners
   */
  async init() {
    this.currentLanguage = this.userSettings.language || 'en';
    this.initClock();
    this.applyTranslations(this.currentLanguage);
    this.setupNavigation();
    this.setupCameraAndCapture();
    this.setupQuestionnaire();
    this.setupLanguageScreen();
    this.setupHistoryScreen();
    this.setupGuidesScreen();
    this.setupSettingsScreen();
    this.setupJotformModal();
    this.setupAudioListeners();
    this.renderDashboardRecent();
    this.registerServiceWorker();

    // Handle Desktop Controls
    const btnToggleFrame = document.getElementById('btn-toggle-frame');
    if (btnToggleFrame) {
      btnToggleFrame.addEventListener('click', () => {
        const frame = document.getElementById('android-device');
        frame.classList.toggle('fullscreen-mode');
      });
    }

    const btnQuickDemo = document.getElementById('btn-quick-sample');
    if (btnQuickDemo) {
      btnQuickDemo.addEventListener('click', () => {
        this.runQuickDemo();
      });
    }

    console.log('RootCause Android App initialized successfully in offline-first mode.');
  }

  /**
   * Keep status bar clock updated
   */
  initClock() {
    const clockElem = document.getElementById('status-clock');
    const updateTime = () => {
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      if (clockElem) clockElem.textContent = `${hours}:${minutes}`;
    };
    updateTime();
    setInterval(updateTime, 30000);
  }

  /**
   * Screen Navigation State Machine
   */
  navigateTo(screenId, addToHistory = true) {
    if (screenId === this.currentScreen) return;

    // Stop camera if navigating away from capture screen
    if (this.currentScreen === 'screen-capture') {
      CameraService.stopCamera();
    }

    // Stop active audio if changing screens
    AudioService.stop();

    if (addToHistory && this.currentScreen !== 'screen-welcome') {
      this.screenHistory.push(this.currentScreen);
    }

    // Hide all screens, show target screen
    document.querySelectorAll('.app-screen').forEach(screen => {
      screen.classList.remove('active');
    });

    const targetScreen = document.getElementById(screenId);
    if (targetScreen) {
      targetScreen.classList.add('active');
      this.currentScreen = screenId;

      // Reset scroll position
      const viewport = document.querySelector('.screens-viewport');
      if (viewport) viewport.scrollTop = 0;
    }

    this.updateHeaderAndNav(screenId);

    // Lifecycle triggers per screen
    if (screenId === 'screen-capture') {
      this.onEnterCaptureScreen();
    } else if (screenId === 'screen-history') {
      this.renderFullHistory();
    } else if (screenId === 'screen-dashboard') {
      this.renderDashboardRecent();
    } else if (screenId === 'screen-guides') {
      this.renderGuidesList('all');
    } else if (screenId === 'screen-field-questionnaire') {
      if (this.currentDiagnosis) {
        this.currentDisease = this.currentDiagnosis.disease;
        this.isViewingCatalogDisease = false;
      }
      this.renderQuestionnaire();
    } else if (screenId === 'screen-questionnaire-consent') {
      if (this.currentDiagnosis) {
        this.currentDisease = this.currentDiagnosis.disease;
        this.isViewingCatalogDisease = false;
      }
      this.renderQuestionnairePrediction('questionnaire-prediction-consent');
    } else if (screenId === 'screen-results' && this.currentDiagnosis) {
      this.currentDisease = this.currentDiagnosis.disease;
      this.isViewingCatalogDisease = false;
      this.renderResultsScreen(this.currentDiagnosis);
    } else if (screenId === 'screen-advisory' && this.currentDisease) {
      this.renderAdvisoryScreen(this.currentDisease);
    }
  }

  goBack() {
    if (this.screenHistory.length > 0) {
      const prevScreen = this.screenHistory.pop();
      this.navigateTo(prevScreen, false);
    } else {
      this.navigateTo('screen-dashboard', false);
    }
  }

  updateHeaderAndNav(screenId) {
    const topBar = document.getElementById('top-bar');
    const topBackBtn = document.getElementById('btn-top-back');
    const topTitle = document.getElementById('top-app-title');

    // Welcome screen doesn't show top bar or back button
    if (screenId === 'screen-welcome') {
      topBar.style.display = 'none';
    } else {
      topBar.style.display = 'flex';
      // Show back button on inner flow screens
      if (['screen-crop-select', 'screen-capture', 'screen-questionnaire-consent', 'screen-field-questionnaire', 'screen-results', 'screen-advisory', 'screen-languages'].includes(screenId)) {
        topBackBtn.style.display = 'flex';
      } else {
        topBackBtn.style.display = 'none';
      }
    }

    // Update screen titles in header
    const titleTextSpan = topTitle.querySelector('span');
    if (titleTextSpan) {
      const titles = {
        'screen-dashboard': 'RootCause',
        'screen-crop-select': getTranslation(this.currentLanguage, 'cropSelectHeading'),
        'screen-capture': getTranslation(this.currentLanguage, 'captureHeading'),
        'screen-questionnaire-consent': 'Field Context',
        'screen-field-questionnaire': 'Field Observations',
        'screen-results': getTranslation(this.currentLanguage, 'resultsHeading'),
        'screen-advisory': getTranslation(this.currentLanguage, 'advisoryHeading'),
        'screen-languages': getTranslation(this.currentLanguage, 'languageSection'),
        'screen-history': getTranslation(this.currentLanguage, 'historyHeading'),
        'screen-settings': getTranslation(this.currentLanguage, 'settingsHeading'),
        'screen-guides': 'Crop Guides'
      };
      titleTextSpan.textContent = titles[screenId] || 'RootCause';
    }

    // Update Bottom Nav active indicator
    document.querySelectorAll('.nav-tab-item').forEach(tab => {
      const targetNav = tab.getAttribute('data-nav');
      if (targetNav === screenId) {
        tab.classList.add('active');
      } else {
        tab.classList.remove('active');
      }
    });
  }

  /**
   * Setup Event Handlers for Navigation & Buttons
   */
  setupNavigation() {
    // Top Bar Back Button
    document.getElementById('btn-top-back').addEventListener('click', () => this.goBack());

    // Top Bar Language Shortcut Button
    document.getElementById('btn-top-lang').addEventListener('click', () => {
      this.navigateTo('screen-languages');
    });

    // Welcome Screen: Get Started button
    document.getElementById('btn-welcome-start').addEventListener('click', () => {
      this.navigateTo('screen-dashboard');
    });

    // Dashboard: Hero Scan button
    document.getElementById('card-hero-detect').addEventListener('click', () => {
      this.navigateTo('screen-crop-select');
    });

    // Dashboard: Crop Quick Cards
    document.getElementById('btn-crop-rice').addEventListener('click', () => {
      this.currentCrop = 'rice';
      this.navigateTo('screen-capture');
    });

    document.getElementById('btn-crop-wheat').addEventListener('click', () => {
      this.currentCrop = 'wheat';
      this.navigateTo('screen-capture');
    });

    // Dashboard: Quick Links
    document.getElementById('link-view-all-guides').addEventListener('click', () => {
      this.navigateTo('screen-guides');
    });

    document.getElementById('link-view-all-history').addEventListener('click', () => {
      this.navigateTo('screen-history');
    });

    // Bottom Navigation Bar tabs
    document.querySelectorAll('.nav-tab-item').forEach(tab => {
      tab.addEventListener('click', () => {
        const targetScreen = tab.getAttribute('data-nav');
        if (targetScreen) this.navigateTo(targetScreen);
      });
    });

    // Crop Selection Screen buttons
    document.getElementById('btn-proceed-rice').addEventListener('click', () => {
      this.currentCrop = 'rice';
      this.navigateTo('screen-capture');
    });

    document.getElementById('btn-guide-rice').addEventListener('click', () => {
      this.navigateTo('screen-guides');
      this.renderGuidesList('rice');
    });

    document.getElementById('btn-proceed-wheat').addEventListener('click', () => {
      this.currentCrop = 'wheat';
      this.navigateTo('screen-capture');
    });

    document.getElementById('btn-guide-wheat').addEventListener('click', () => {
      this.navigateTo('screen-guides');
      this.renderGuidesList('wheat');
    });

    // Capture Screen: Change Crop button
    document.getElementById('btn-change-crop').addEventListener('click', () => {
      this.navigateTo('screen-crop-select');
    });

    // Results Screen buttons
    document.getElementById('btn-results-view-advisory').addEventListener('click', () => {
      if (this.currentDisease) {
        this.navigateTo('screen-advisory');
      }
    });

    document.getElementById('btn-results-save-history').addEventListener('click', () => {
      if (this.currentDiagnosis) {
        const savedDiagnosis = {
          ...this.currentDiagnosis,
          questionnaireSessionId: this.questionnaireSession?.id || null,
          fieldObservations: this.questionnaireSession?.answers || null,
          contextualAssessment: this.questionnaireSession?.assessment || null
        };
        StorageService.saveDiagnosis(savedDiagnosis);
        alert('Diagnosis successfully saved to local history!');
      }
    });

    document.getElementById('btn-results-new-scan').addEventListener('click', () => {
      this.navigateTo('screen-capture');
    });

    document.getElementById('btn-quality-retake').addEventListener('click', () => {
      this.navigateTo('screen-capture');
    });
    document.getElementById('btn-unavailable-retry').addEventListener('click', () => {
      this.navigateTo('screen-capture');
    });

    document.getElementById('btn-inconclusive-ask-expert').addEventListener('click', () => {
      this.openJotformModal();
    });

    // Advisory Screen: Scan again
    document.getElementById('btn-advisory-scan-again').addEventListener('click', () => {
      this.navigateTo('screen-capture');
    });
  }

  /**
   * Setup Camera & Image Capture Viewport
   */
  setupCameraAndCapture() {
    const tabCamera = document.getElementById('tab-camera');
    const tabGallery = document.getElementById('tab-gallery');
    const cameraBox = document.getElementById('camera-viewport-box');
    const galleryBox = document.getElementById('gallery-upload-box');
    const previewBox = document.getElementById('image-preview-box');
    const cameraControls = document.getElementById('camera-controls-bar');
    const videoElem = document.getElementById('camera-video-elem');
    const fileInput = document.getElementById('file-leaf-input');
    const btnBrowse = document.getElementById('btn-browse-files');
    const btnSnap = document.getElementById('btn-snap-photo');
    const btnRetake = document.getElementById('btn-retake-photo');
    const btnRemove = document.getElementById('btn-remove-photo');
    const btnRun = document.getElementById('btn-run-diagnosis');

    // Tab: Camera Mode
    tabCamera.addEventListener('click', async () => {
      tabCamera.classList.add('active');
      tabGallery.classList.remove('active');
      galleryBox.style.display = 'none';

      if (!this.currentImageData) {
        cameraBox.style.display = 'block';
        cameraControls.style.display = 'block';
        previewBox.style.display = 'none';
        await CameraService.startCamera(videoElem);
      }
    });

    // Tab: Gallery Mode
    tabGallery.addEventListener('click', () => {
      tabGallery.classList.add('active');
      tabCamera.classList.remove('active');
      CameraService.stopCamera();
      cameraBox.style.display = 'none';
      cameraControls.style.display = 'none';

      if (!this.currentImageData) {
        galleryBox.style.display = 'flex';
        previewBox.style.display = 'none';
      }
    });

    // Snap Photo
    btnSnap.addEventListener('click', () => {
      let capturedData = CameraService.captureFrame(videoElem);
      if (!capturedData) {
        // In case camera was simulated or denied, fall back to crop sample image
        capturedData = this.SAMPLE_IMAGES[this.currentCrop];
      }
      this.setImagePreview(capturedData);
    });

    // Gallery Browse
    btnBrowse.addEventListener('click', () => fileInput.click());
    galleryBox.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (file) {
        try {
          const optimizedData = await CameraService.processImageFile(file);
          this.setImagePreview(optimizedData);
        } catch (err) {
          alert('Could not load image: ' + err.message);
        }
      }
    });

    // Retake / Remove Photo
    btnRetake.addEventListener('click', () => this.resetCaptureView());
    btnRemove.addEventListener('click', () => this.resetCaptureView());

    // Toggle Field Details accordion
    const toggleDetails = document.getElementById('toggle-field-details');
    const detailsContainer = document.getElementById('field-details-container');
    const detailsArrow = document.getElementById('field-details-arrow');
    toggleDetails.addEventListener('click', () => {
      const isOpen = detailsContainer.classList.toggle('open');
      detailsArrow.textContent = isOpen ? '▲' : '▼';
    });

    // Run Disease Detection
    btnRun.addEventListener('click', async () => {
      await this.executeDiagnosis();
    });

  }

  onEnterCaptureScreen() {
    // A new scan starts a new diagnosis session and cannot inherit old answers.
    this.currentDiagnosis = null;
    this.currentDisease = null;
    this.questionnaireSession = null;
    this.isViewingCatalogDisease = false;
    const notes = document.getElementById('input-farmer-notes');
    if (notes) notes.value = '';
    // A capture screen always starts a new scan. Clear any sample or previous
    // upload so the gallery can be opened and a fresh image can be selected.
    this.resetCaptureView();

    // Update Crop Indicator
    const cropLabel = document.getElementById('capture-selected-crop');
    if (cropLabel) {
      cropLabel.textContent = this.currentCrop === 'rice' ? 'Rice Paddy (धान)' : 'Wheat (गेहूं)';
    }
  }

  setImagePreview(dataUrl) {
    this.currentImageData = dataUrl;
    CameraService.stopCamera();

    const cameraBox = document.getElementById('camera-viewport-box');
    const galleryBox = document.getElementById('gallery-upload-box');
    const previewBox = document.getElementById('image-preview-box');
    const previewImg = document.getElementById('leaf-preview-img');
    const cameraControls = document.getElementById('camera-controls-bar');
    const btnRun = document.getElementById('btn-run-diagnosis');

    cameraBox.style.display = 'none';
    galleryBox.style.display = 'none';
    cameraControls.style.display = 'none';

    previewImg.src = dataUrl;
    previewBox.style.display = 'flex';
    btnRun.disabled = false;
  }

  resetCaptureView() {
    this.currentImageData = null;
    const previewBox = document.getElementById('image-preview-box');
    const previewImg = document.getElementById('leaf-preview-img');
    const btnRun = document.getElementById('btn-run-diagnosis');
    const tabCamera = document.getElementById('tab-camera');
    const videoElem = document.getElementById('camera-video-elem');
    const fileInput = document.getElementById('file-leaf-input');

    previewImg.src = '';
    previewBox.style.display = 'none';
    btnRun.disabled = true;
    fileInput.value = '';

    if (tabCamera.classList.contains('active')) {
      document.getElementById('camera-viewport-box').style.display = 'block';
      document.getElementById('camera-controls-bar').style.display = 'block';
      CameraService.startCamera(videoElem);
    } else {
      document.getElementById('gallery-upload-box').style.display = 'flex';
    }
  }

  /**
   * Run disease detection through the FastAPI backend
   */
  async executeDiagnosis() {
    const btnRun = document.getElementById('btn-run-diagnosis');
    const originalText = btnRun.innerHTML;
    btnRun.disabled = true;
    btnRun.innerHTML = `<span class="spinner" style="width: 18px; height: 18px; border-width: 2px; margin: 0; display: inline-block;"></span> Analyzing Leaf...`;

    try {
      const result = await ClassifierService.classifyLeaf({
        imageData: this.currentImageData || this.SAMPLE_IMAGES[this.currentCrop],
        crop: this.currentCrop
      });

      // Attach optional field notes from the form
      const firstName = document.getElementById('input-farmer-first').value.trim();
      const lastName = document.getElementById('input-farmer-last').value.trim();
      const region = document.getElementById('input-farmer-region').value.trim();
      const notes = document.getElementById('input-farmer-notes').value.trim();

      // Keep model output fields as returned by ClassifierService. Farmer and
      // questionnaire information are attached as separate session metadata.
      this.currentDiagnosis = Object.freeze({
        ...result,
        farmerName: (firstName || lastName) ? `${firstName} ${lastName}`.trim() : 'Local Farmer',
        region: region || 'Farm Field',
        notes,
        imageThumbnail: this.currentImageData || this.SAMPLE_IMAGES[this.currentCrop]
      });
      this.currentDisease = result.disease;
      this.isViewingCatalogDisease = false;
      this.questionnaireSession = {
        id: `diagnosis_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        crop: result.crop || this.currentCrop,
        diseaseId: this.currentDisease?.id || null,
        answers: null,
        assessment: null,
        skipped: false
      };

      if (result.isBackend && !result.isUnavailable && !result.isQualityFailed && result.disease) {
        this.navigateTo('screen-questionnaire-consent');
      } else {
        this.navigateTo('screen-results');
      }
    } catch (err) {
      alert('Diagnosis failed: ' + err.message);
    } finally {
      btnRun.disabled = false;
      btnRun.innerHTML = originalText;
    }
  }

  /**
   * Render Screen 5: Disease Detection Results
   */
  renderResultsScreen(result) {
    const hero = document.querySelector('#screen-results .results-card-hero');
    const qualityBox = document.getElementById('results-quality-box');
    const unavailableBox = document.getElementById('results-unavailable-box');
    const inconclusiveBox = document.getElementById('results-inconclusive-box');
    const btnAdvisory = document.getElementById('btn-results-view-advisory');
    const confidenceBox = document.querySelector('.confidence-meter-box');
    const actions = document.querySelector('#screen-results > div[style*="flex-direction: column"]');
    document.getElementById('results-context-section').style.display = 'none';
    qualityBox.style.display = result.isQualityFailed ? 'block' : 'none';
    unavailableBox.style.display = result.isUnavailable ? 'block' : 'none';
    unavailableBox.querySelector('#results-unavailable-message').textContent = result.message || 'Prediction service unavailable.';
    if (result.isQualityFailed) {
      const issueList = document.getElementById('results-quality-issues');
      issueList.replaceChildren(...(result.issues || []).map(issue => {
        const item = document.createElement('li');
        item.textContent = issue;
        return item;
      }));
      document.getElementById('results-quality-suggestions').textContent = (result.suggestions || []).join(' ');
    }
    hero.style.display = result.isUnavailable || result.isQualityFailed ? 'none' : '';
    confidenceBox.style.display = result.isBackend && !result.isQualityFailed ? '' : 'none';
    inconclusiveBox.style.display = 'none';
    btnAdvisory.style.display = 'none';
    if (actions) actions.style.display = result.isUnavailable || result.isQualityFailed ? 'none' : '';
    if (result.isUnavailable || result.isQualityFailed) return;

    this.renderFieldContext('results-context-section', 'results-observation-list', 'results-context-assessment');

    const leafImg = document.getElementById('results-leaf-img');
    const cropTag = document.getElementById('results-crop-tag');
    const diseaseName = document.getElementById('results-disease-name');
    const sciName = document.getElementById('results-scientific-name');
    const confScore = document.getElementById('results-conf-score');
    const confBar = document.getElementById('results-conf-bar');
    const severityBadge = document.getElementById('results-severity-badge');
    const summaryText = document.getElementById('results-summary-text');

    leafImg.src = result.imageThumbnail || this.SAMPLE_IMAGES[result.crop];
    cropTag.textContent = result.cropDisplayName;

    if (result.isInconclusive) {
      diseaseName.textContent = 'Inconclusive Symptoms';
      sciName.textContent = 'Uncertain Classification';
      confScore.textContent = `${result.confidence}%`;
      confBar.style.width = `${result.confidence}%`;
      confBar.style.background = '#ef4444';
      severityBadge.className = 'badge badge-critical';
      severityBadge.textContent = 'Low Confidence';
      summaryText.textContent = result.message;

      inconclusiveBox.style.display = 'block';
      btnAdvisory.style.display = 'none';
    } else {
      diseaseName.textContent = result.diseaseName;
      sciName.textContent = result.scientificName;
      confScore.textContent = `${result.confidence}%`;
      confBar.style.width = `${result.confidence}%`;
      confBar.style.background = 'linear-gradient(90deg, #22c55e 0%, #15803d 100%)';

      severityBadge.className = `badge ${result.badgeClass || 'badge-high'}`;
      severityBadge.textContent = `${result.severity} Priority`;
      summaryText.textContent = result.basicSymptoms;

      const sourcePill = document.getElementById('results-source-pill');
      sourcePill.className = 'backend-source-pill live';
      sourcePill.textContent = `🟢 ${result.backendSource || 'FastAPI model'}`;

      inconclusiveBox.style.display = 'none';
      btnAdvisory.style.display = 'flex';
    }
  }

  /**
   * Render Screen 6: Disease Information and Advisory
   */
  renderAdvisoryScreen(disease) {
    if (!disease) return;

    document.getElementById('advisory-crop-badge').textContent = disease.cropDisplayName;
    document.getElementById('advisory-disease-name').textContent = disease.name;
    document.getElementById('advisory-sci-name').textContent = disease.scientificName;
    
    const severityBadge = document.getElementById('advisory-severity-badge');
    severityBadge.className = `badge ${disease.badgeClass || 'badge-high'}`;
    severityBadge.textContent = `${disease.severity} Priority`;

    document.getElementById('advisory-overview-text').textContent = disease.summary;
    document.getElementById('advisory-signs-text').textContent = disease.visibleSigns;

    // Render Symptoms list
    const symptomsList = document.getElementById('advisory-symptoms-list');
    symptomsList.innerHTML = disease.symptoms.map(s => `<li>${s}</li>`).join('');

    // Render Causes text
    document.getElementById('advisory-causes-text').textContent = disease.causes;

    // Render Organic Treatment list
    const organicList = document.getElementById('advisory-organic-list');
    organicList.innerHTML = disease.treatment.organic.map(t => `<li>${t}</li>`).join('');

    // Render Chemical Treatment list
    const chemicalList = document.getElementById('advisory-chemical-list');
    chemicalList.innerHTML = disease.treatment.chemical.map(t => `<li>${t}</li>`).join('');

    // Render Prevention list
    const preventionList = document.getElementById('advisory-prevention-list');
    preventionList.innerHTML = disease.prevention.map(p => `<li>${p}</li>`).join('');

    // Expert warning
    document.getElementById('advisory-warning-text').textContent = disease.expertWarning;
    this.renderFieldContext('advisory-field-context', 'advisory-observation-list', 'advisory-context-assessment');
  }

  setupQuestionnaire() {
    document.getElementById('btn-questionnaire-yes').addEventListener('click', () => {
      if (!this.questionnaireSession) return;
      this.navigateTo('screen-field-questionnaire');
    });
    document.getElementById('btn-questionnaire-no').addEventListener('click', () => {
      if (this.questionnaireSession) {
        this.questionnaireSession = { ...this.questionnaireSession, answers: null, assessment: null, skipped: true };
      }
      this.navigateTo('screen-results');
    });
    document.getElementById('btn-questionnaire-cancel').addEventListener('click', () => this.goBack());

    const form = document.getElementById('field-questionnaire-form');
    form.addEventListener('input', () => this.saveQuestionnaireDraft());
    form.addEventListener('change', () => this.saveQuestionnaireDraft());
    form.addEventListener('submit', event => {
      event.preventDefault();
      if (!this.questionnaireSession || !this.currentDisease || !this.currentDiagnosis) return;
      this.saveQuestionnaireDraft();
      const answers = this.questionnaireSession.answers || {};
      this.questionnaireSession = {
        ...this.questionnaireSession,
        assessment: assessFieldObservations(this.currentDisease, answers),
        skipped: false
      };
      this.navigateTo('screen-results');
    });
  }

  renderQuestionnaire() {
    const session = this.questionnaireSession;
    if (!session) return;
    const config = getQuestionnaireOptions(session.crop);
    const container = document.getElementById('questionnaire-fields');
    this.renderQuestionnairePrediction('questionnaire-prediction-form');
    document.getElementById('questionnaire-heading').textContent = `${session.crop === 'rice' ? 'Rice' : 'Wheat'} field observations`;
    container.replaceChildren();
    if (!config) return;

    const answers = session.answers || {};
    const fieldset = document.createElement('fieldset');
    fieldset.className = 'questionnaire-fieldset';
    const legend = document.createElement('legend');
    legend.textContent = 'What symptoms do you see? Select all that apply.';
    fieldset.append(legend);
    config.symptoms.forEach(([value, label]) => {
      const row = document.createElement('label');
      row.className = 'questionnaire-choice';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.name = 'symptoms';
      input.value = value;
      input.checked = (answers.symptoms || []).includes(value);
      const text = document.createElement('span');
      text.textContent = label;
      row.append(input, text);
      fieldset.append(row);
    });
    container.append(fieldset);

    const selectQuestion = (key, title, choices) => {
      const group = document.createElement('div');
      group.className = 'form-group questionnaire-select-group';
      const label = document.createElement('label');
      label.className = 'form-label';
      label.htmlFor = `questionnaire-${key}`;
      label.textContent = title;
      const select = document.createElement('select');
      select.className = 'form-select';
      select.id = `questionnaire-${key}`;
      select.name = key;
      choices.forEach(([value, text]) => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        select.append(option);
      });
      select.value = answers[key] || 'unsure';
      group.append(label, select);
      container.append(group);
    };
    selectQuestion('affectedArea', 'How much of the crop area is affected?', [['unsure','Not sure'],['under_5','Less than 5%'],['5_25','About 5–25%'],['26_50','About 26–50%'],['over_50','More than 50%']]);
    selectQuestion('growthStage', 'What is the crop growth stage?', [['unsure','Not sure'],['seedling','Seedling'],['vegetative','Vegetative / tillering'],['flowering','Flowering / heading'],['grain_fill','Grain filling / nearing harvest']]);
    selectQuestion('duration', 'How long have you noticed the symptoms?', [['unsure','Not sure'],['under_3_days','Less than 3 days'],['3_7_days','3–7 days'],['1_3_weeks','1–3 weeks'],['over_3_weeks','More than 3 weeks']]);
    selectQuestion('spread', 'Are symptoms spreading to more leaves or plants?', [['unsure','Not sure'],['yes','Yes'],['no','No']]);
    selectQuestion('weather', 'Recent weather around the field', [['unsure','Not sure'],['rain_humid','Recent rain or humid conditions'],['dry','Mostly dry conditions']]);
    selectQuestion('irrigation', session.crop === 'rice' ? 'Recent irrigation or field water conditions' : 'Recent wheat irrigation', config.irrigation);
  }

  renderQuestionnairePrediction(elementId) {
    const container = document.getElementById(elementId);
    const diagnosis = this.currentDiagnosis;
    container.replaceChildren();
    if (!diagnosis) return;
    const crop = document.createElement('div');
    crop.textContent = `Crop: ${diagnosis.cropDisplayName || diagnosis.crop}`;
    const prediction = document.createElement('div');
    prediction.textContent = `AI image prediction: ${diagnosis.diseaseName || diagnosis.prediction}`;
    const confidence = document.createElement('div');
    confidence.textContent = `Original model confidence: ${diagnosis.confidence}%`;
    container.append(crop, prediction, confidence);
  }

  saveQuestionnaireDraft() {
    const session = this.questionnaireSession;
    if (!session || this.currentScreen !== 'screen-field-questionnaire') return;
    const form = document.getElementById('field-questionnaire-form');
    const data = new FormData(form);
    session.answers = {
      symptoms: data.getAll('symptoms'),
      affectedArea: data.get('affectedArea') || 'unsure',
      growthStage: data.get('growthStage') || 'unsure',
      duration: data.get('duration') || 'unsure',
      spread: data.get('spread') || 'unsure',
      weather: data.get('weather') || 'unsure',
      irrigation: data.get('irrigation') || 'unsure'
    };
  }

  renderFieldContext(sectionId, listId, assessmentId) {
    const section = document.getElementById(sectionId);
    const list = document.getElementById(listId);
    const assessmentElement = document.getElementById(assessmentId);
    const session = this.questionnaireSession;
    if (!session?.answers || session.skipped || this.isViewingCatalogDisease || !this.currentDisease || session.diseaseId !== this.currentDisease.id) {
      section.style.display = 'none';
      return;
    }
    const labels = this.getObservationLabels(session.crop, session.answers);
    if (this.currentDiagnosis?.notes) labels.push(['Earlier field notes', this.currentDiagnosis.notes]);
    list.replaceChildren();
    if (labels.length === 0) {
      const item = document.createElement('dd');
      item.textContent = 'No field answers were provided.';
      list.append(item);
    } else {
      labels.forEach(([label, value]) => {
        const term = document.createElement('dt');
        term.textContent = label;
        const description = document.createElement('dd');
        description.textContent = value;
        list.append(term, description);
      });
    }
    const assessment = session.assessment || assessFieldObservations(this.currentDisease, session.answers);
    assessmentElement.replaceChildren();
    const heading = document.createElement('strong');
    heading.textContent = assessment.title;
    const explanation = document.createElement('p');
    explanation.textContent = assessment.explanation;
    assessmentElement.dataset.status = assessment.status;
    assessmentElement.append(heading, explanation);
    section.style.display = 'block';
  }

  getObservationLabels(crop, answers) {
    const config = getQuestionnaireOptions(crop);
    if (!config) return [];
    const output = [];
    const selected = (answers.symptoms || []).map(value => config.symptoms.find(item => item[0] === value)?.[1]).filter(Boolean);
    if (selected.length) output.push(['Visible symptoms', selected.join('; ')]);
    const fields = [
      ['affectedArea', 'Affected area', {under_5:'Less than 5%', '5_25':'About 5–25%', '26_50':'About 26–50%', over_50:'More than 50%'}],
      ['growthStage', 'Growth stage', {seedling:'Seedling', vegetative:'Vegetative / tillering', flowering:'Flowering / heading', grain_fill:'Grain filling / nearing harvest'}],
      ['duration', 'Symptom duration', {under_3_days:'Less than 3 days', '3_7_days':'3–7 days', '1_3_weeks':'1–3 weeks', over_3_weeks:'More than 3 weeks'}],
      ['spread', 'Spreading', {yes:'Yes', no:'No'}],
      ['weather', 'Recent weather', {rain_humid:'Recent rain or humid conditions', dry:'Mostly dry conditions'}],
      ['irrigation', 'Irrigation', Object.fromEntries(config.irrigation.map(([value,label]) => [value,label]))]
    ];
    fields.forEach(([key,label,map]) => { if (answers[key] && answers[key] !== 'unsure' && map[answers[key]]) output.push([label,map[answers[key]]]); });
    return output;
  }

  /**
   * Setup Audio / Speech Synthesis Listeners
   */
  setupAudioListeners() {
    const btnAudioToggle = document.getElementById('btn-audio-toggle');
    const audioBtnIcon = document.getElementById('audio-btn-icon');
    const audioWaves = document.getElementById('audio-waves');
    const audioStatusLabel = document.getElementById('audio-status-label');

    btnAudioToggle.addEventListener('click', () => {
      if (AudioService.isSpeaking) {
        AudioService.stop();
      } else {
        if (this.currentDisease) {
          AudioService.speakDiseaseAdvisory(this.currentDisease, this.currentLanguage);
        }
      }
    });

    AudioService.setListener((isSpeaking) => {
      if (isSpeaking) {
        audioBtnIcon.textContent = '⏹';
        audioWaves.style.display = 'flex';
        audioStatusLabel.textContent = getTranslation(this.currentLanguage, 'stopVoice');
      } else {
        audioBtnIcon.textContent = '🔊';
        audioWaves.style.display = 'none';
        audioStatusLabel.textContent = getTranslation(this.currentLanguage, 'listenVoice');
      }
    });
  }

  /**
   * Setup Screen 7: Language Settings
   */
  setupLanguageScreen() {
    const listContainer = document.getElementById('language-list-container');
    listContainer.innerHTML = SUPPORTED_LANGUAGES.map(lang => `
      <div class="lang-card-item ${lang.code === this.currentLanguage ? 'selected' : ''}" data-lang-code="${lang.code}">
        <div class="lang-card-left">
          <span class="lang-flag">${lang.flag}</span>
          <div>
            <div class="lang-name-main">${lang.native}</div>
            <div class="lang-name-sub">${lang.name}</div>
          </div>
        </div>
        <div class="lang-radio"></div>
      </div>
    `).join('');

    listContainer.querySelectorAll('.lang-card-item').forEach(item => {
      item.addEventListener('click', () => {
        listContainer.querySelectorAll('.lang-card-item').forEach(i => i.classList.remove('selected'));
        item.classList.add('selected');
        const selectedCode = item.getAttribute('data-lang-code');
        this.selectLanguage(selectedCode);
      });
    });

    document.getElementById('btn-apply-language').addEventListener('click', () => {
      this.goBack();
    });
  }

  selectLanguage(langCode) {
    this.currentLanguage = langCode;
    StorageService.saveSettings({ language: langCode });
    this.applyTranslations(langCode);

    const topCode = document.getElementById('top-lang-code');
    if (topCode) topCode.textContent = langCode.toUpperCase();

    const settingsLangName = document.getElementById('settings-current-lang-name');
    const matched = SUPPORTED_LANGUAGES.find(l => l.code === langCode);
    if (settingsLangName && matched) {
      settingsLangName.textContent = `${matched.native} (${matched.name})`;
    }
  }

  applyTranslations(langCode) {
    document.querySelectorAll('[data-i18n]').forEach(elem => {
      const key = elem.getAttribute('data-i18n');
      const text = getTranslation(langCode, key);
      if (text) {
        elem.textContent = text;
      }
    });
  }

  /**
   * Setup Screen 8: Diagnosis History
   */
  setupHistoryScreen() {
    // Filter chips
    document.querySelectorAll('.history-filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.history-filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const filter = chip.getAttribute('data-filter');
        this.renderFullHistory(filter);
      });
    });

    // Clear history button
    document.getElementById('btn-clear-history-records').addEventListener('click', () => {
      if (confirm('Are you sure you want to clear your local diagnosis history?')) {
        StorageService.clearHistory();
        this.renderFullHistory();
        this.renderDashboardRecent();
      }
    });
  }

  renderFullHistory(cropFilter = 'all') {
    const container = document.getElementById('history-full-list');
    let history = StorageService.getHistory();

    if (cropFilter !== 'all') {
      history = history.filter(h => h.crop === cropFilter);
    }

    if (history.length === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 40px 20px; color: var(--color-text-muted);">
          <div style="font-size: 38px; margin-bottom: 8px;">📋</div>
          <div style="font-weight: 600;">${getTranslation(this.currentLanguage, 'emptyHistory')}</div>
        </div>
      `;
      return;
    }

    container.innerHTML = history.map(item => `
      <div class="history-full-card" data-diag-id="${item.id}">
        <div class="history-card-thumb">
          <img src="${item.imageThumbnail || this.SAMPLE_IMAGES[item.crop]}" alt="Leaf thumb">
        </div>
        <div class="history-card-body">
          <div class="history-card-top-row">
            <span class="crop-chip" style="font-size: 10.5px;">${item.cropDisplayName || item.crop}</span>
            <span class="badge ${item.severity === 'Critical' ? 'badge-critical' : 'badge-high'}">${item.confidence}% Match</span>
          </div>
          <div class="history-disease-title">${item.diseaseName}</div>
          <div class="history-date-time">${item.displayDate || item.timestamp}</div>
          <div style="font-size: 11.5px; color: var(--color-text-muted); line-height: 1.3;">
            ${item.region ? `📍 ${item.region}` : ''} ${item.notes ? `• ${item.notes}` : ''}
          </div>
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.history-full-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-diag-id');
        const record = history.find(h => h.id === id);
        if (record) {
          const disease = getDiseaseById(record.diseaseId);
          if (disease) {
            this.isViewingCatalogDisease = true;
            this.currentDisease = disease;
            this.renderAdvisoryScreen(disease);
            this.navigateTo('screen-advisory');
          }
        }
      });
    });
  }

  renderDashboardRecent() {
    const container = document.getElementById('dashboard-recent-list');
    const history = StorageService.getHistory().slice(0, 3);

    if (history.length === 0) {
      container.innerHTML = `
        <div style="text-align: center; padding: 18px; color: var(--color-text-muted); font-size: 13px;">
          ${getTranslation(this.currentLanguage, 'noRecent')}
        </div>
      `;
      return;
    }

    container.innerHTML = history.map(item => `
      <div class="recent-history-item" data-diag-id="${item.id}">
        <div class="recent-item-thumb">
          <img src="${item.imageThumbnail || this.SAMPLE_IMAGES[item.crop]}" alt="Leaf">
        </div>
        <div class="recent-item-details">
          <div class="recent-item-name">${item.diseaseName}</div>
          <div class="recent-item-meta">
            <span>${item.cropDisplayName || item.crop}</span>
            <span>•</span>
            <span>${item.displayDate}</span>
          </div>
        </div>
        <div class="recent-item-conf">${item.confidence}%</div>
      </div>
    `).join('');

    container.querySelectorAll('.recent-history-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.getAttribute('data-diag-id');
        const historyList = StorageService.getHistory();
        const record = historyList.find(h => h.id === id);
        if (record) {
          const disease = getDiseaseById(record.diseaseId);
          if (disease) {
            this.isViewingCatalogDisease = true;
            this.currentDisease = disease;
            this.renderAdvisoryScreen(disease);
            this.navigateTo('screen-advisory');
          }
        }
      });
    });
  }

  /**
   * Setup Guides Screen Catalog (Allows browsing all 9 diseases)
   */
  setupGuidesScreen() {
    document.querySelectorAll('[data-guide-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-guide-filter]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const filter = btn.getAttribute('data-guide-filter');
        this.renderGuidesList(filter);
      });
    });
  }

  renderGuidesList(crop = 'all') {
    const container = document.getElementById('guides-list-container');
    let diseases = getAllDiseases();

    if (crop !== 'all') {
      diseases = getDiseasesByCrop(crop);
    }

    container.innerHTML = diseases.map(d => `
      <div class="recent-history-item" data-disease-id="${d.id}" style="padding: 14px 16px;">
        <div class="recent-item-thumb" style="background-color: var(--color-surface-variant); display: flex; align-items: center; justify-content: center; font-size: 22px;">
          ${d.crop === 'rice' ? '🌾' : '🌱'}
        </div>
        <div class="recent-item-details">
          <div class="recent-item-name" style="font-size: 15px;">${d.name}</div>
          <div class="recent-item-meta">
            <span>${d.cropDisplayName}</span>
            <span>•</span>
            <span style="font-style: italic;">${d.scientificName}</span>
          </div>
          <div style="font-size: 11px; color: var(--color-text-subtle); margin-top: 3px;">
            Season: ${d.season}
          </div>
        </div>
        <span class="badge ${d.badgeClass}">${d.severity}</span>
      </div>
    `).join('');

    container.querySelectorAll('.recent-history-item').forEach(item => {
      item.addEventListener('click', () => {
        const id = item.getAttribute('data-disease-id');
        const disease = getDiseaseById(id);
        if (disease) {
          this.isViewingCatalogDisease = true;
          this.currentDisease = disease;
          this.renderAdvisoryScreen(disease);
          this.navigateTo('screen-advisory');
        }
      });
    });
  }

  /**
   * Setup Screen 9: Settings
   */
  setupSettingsScreen() {
    // Language shortcut
    document.getElementById('setting-item-lang').addEventListener('click', () => {
      this.navigateTo('screen-languages');
    });

    // Voice assistance toggle
    const voiceSwitch = document.getElementById('switch-voice');
    voiceSwitch.checked = this.userSettings.voiceEnabled !== false;
    voiceSwitch.addEventListener('change', (e) => {
      StorageService.saveSettings({ voiceEnabled: e.target.checked });
    });

    // Voice speed selector
    const speedSelect = document.getElementById('select-voice-speed');
    speedSelect.value = String(this.userSettings.voiceSpeed || '1.0');
    speedSelect.addEventListener('change', (e) => {
      StorageService.saveSettings({ voiceSpeed: parseFloat(e.target.value) });
    });

    // ML Backend Modal trigger
    document.getElementById('setting-item-ml').addEventListener('click', () => {
      const modal = document.getElementById('modal-ml-config');
      const inputEndpoint = document.getElementById('input-ml-endpoint');
      inputEndpoint.value = this.userSettings.apiEndpoint || '';
      modal.classList.add('open');
    });

    document.getElementById('btn-close-ml-modal').addEventListener('click', () => {
      document.getElementById('modal-ml-config').classList.remove('open');
    });

    document.getElementById('btn-test-ml-health').addEventListener('click', async () => {
      const status = document.getElementById('ml-health-status');
      const input = document.getElementById('input-ml-endpoint');
      const savedEndpoint = this.userSettings.apiEndpoint;
      StorageService.saveSettings({ apiEndpoint: input.value.trim() });
      status.textContent = 'Checking prediction server…';
      const health = await ClassifierService.checkBackendHealth();
      StorageService.saveSettings({ apiEndpoint: savedEndpoint || '' });
      this.userSettings = StorageService.getSettings();
      status.textContent = health.online
        ? `Connected (${health.latencyMs} ms)`
        : `Could not connect to ${health.url}. Check the server URL and CORS settings.`;
    });

    document.getElementById('btn-save-ml-config').addEventListener('click', () => {
      const val = document.getElementById('input-ml-endpoint').value.trim();
      StorageService.saveSettings({ apiEndpoint: val });
      this.userSettings = StorageService.getSettings();

      const statusElem = document.getElementById('settings-ml-endpoint-status');
      if (val) {
        statusElem.textContent = `Custom Endpoint: ${val}`;
      } else {
        statusElem.textContent = `Status: Offline guides available; prediction server not configured`;
      }
      document.getElementById('modal-ml-config').classList.remove('open');
      alert('Machine learning configuration updated successfully.');
    });

    document.getElementById('btn-reset-ml-config').addEventListener('click', () => {
      document.getElementById('input-ml-endpoint').value = '';
      StorageService.saveSettings({ apiEndpoint: '' });
      this.userSettings = StorageService.getSettings();
      document.getElementById('settings-ml-endpoint-status').textContent = `Status: Offline guides available; prediction server not configured`;
      document.getElementById('modal-ml-config').classList.remove('open');
    });
  }

  /**
   * Setup Jotform "Ask a Farming Question" Modal
   */
  setupJotformModal() {
    const modal = document.getElementById('modal-jotform');
    const openBtn = document.getElementById('btn-open-jotform-question');
    const closeBtn = document.getElementById('btn-close-jotform-modal');
    const form = document.getElementById('jotform-query-form');

    if (openBtn) {
      openBtn.addEventListener('click', () => this.openJotformModal());
    }

    if (closeBtn) {
      closeBtn.addEventListener('click', () => modal.classList.remove('open'));
    }

    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        const submission = {
          firstName: document.getElementById('jf-firstname').value.trim(),
          lastName: document.getElementById('jf-lastname').value.trim(),
          email: document.getElementById('jf-email').value.trim(),
          crop: document.getElementById('jf-crop').value,
          region: document.getElementById('jf-region').value.trim(),
          question: document.getElementById('jf-question').value.trim(),
          imageData: this.currentImageData || this.SAMPLE_IMAGES[this.currentCrop]
        };

        StorageService.saveQuestionSubmission(submission);
        modal.classList.remove('open');
        form.reset();
        alert('Thank you, Kisan Mitra! Your question and leaf image have been queued offline and submitted for agricultural extension advisory.');
      });
    }
  }

  openJotformModal() {
    const modal = document.getElementById('modal-jotform');
    const previewImg = document.getElementById('jf-preview-img');
    const cropSelect = document.getElementById('jf-crop');

    if (this.currentImageData) {
      previewImg.src = this.currentImageData;
    } else {
      previewImg.src = this.SAMPLE_IMAGES[this.currentCrop];
    }

    if (this.currentCrop === 'wheat') {
      cropSelect.value = 'Wheat';
    } else {
      cropSelect.value = 'Rice (Paddy)';
    }

    modal.classList.add('open');
  }

  /**
   * Quick Demo Trigger for Evaluators
   */
  async runQuickDemo() {
    this.currentCrop = 'rice';
    this.currentImageData = this.SAMPLE_IMAGES.rice;
    await this.executeDiagnosis();
  }

  /**
   * Register Service Worker for 100% Offline Progressive Web App
   */
  registerServiceWorker() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('sw.js')
        .then(() => console.log('RootCause offline Service Worker registered successfully.'))
        .catch(err => console.log('Service Worker registration skipped:', err));
    }
  }
}

// Bootstrap application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new RootCauseApp();
  app.init();
  window.rootCauseApp = app; // Accessible for debugging
});

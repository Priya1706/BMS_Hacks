/**
 * RootCause Voice Assistance & Audio Service
 * Enables speech synthesis reading of disease advisory in English & Indian languages
 */

import { StorageService } from './storage.js';

export class AudioService {
  static synth = typeof window !== 'undefined' ? window.speechSynthesis : null;
  static currentUtterance = null;
  static isSpeaking = false;
  static onStateChange = null;

  /**
   * Language to Voice locale mapping
   */
  static LOCALE_MAP = {
    en: 'en-IN',
    hi: 'hi-IN',
    kn: 'kn-IN',
    te: 'te-IN',
    ta: 'ta-IN',
    bn: 'bn-IN',
    mr: 'mr-IN',
    pa: 'pa-IN'
  };

  /**
   * Speak a text string in the target language
   */
  static speak(text, language = 'en', onEndCallback = null) {
    if (!this.synth) {
      console.warn('Speech synthesis not supported on this device/browser.');
      return false;
    }

    const settings = StorageService.getSettings();
    if (!settings.voiceEnabled) {
      console.log('Voice assistance is disabled in user settings.');
      return false;
    }

    // Cancel any active speech
    this.stop();

    const utterance = new SpeechSynthesisUtterance(text);
    const targetLocale = this.LOCALE_MAP[language] || 'en-IN';
    utterance.lang = targetLocale;
    utterance.rate = settings.voiceSpeed || 1.0;
    utterance.pitch = 1.0;

    // Try to find matching voice for target locale
    const voices = this.synth.getVoices();
    const matchedVoice = voices.find(v => v.lang.startsWith(targetLocale) || v.lang.startsWith(language));
    if (matchedVoice) {
      utterance.voice = matchedVoice;
    }

    utterance.onstart = () => {
      this.isSpeaking = true;
      if (this.onStateChange) this.onStateChange(true);
    };

    utterance.onend = () => {
      this.isSpeaking = false;
      this.currentUtterance = null;
      if (this.onStateChange) this.onStateChange(false);
      if (onEndCallback) onEndCallback();
    };

    utterance.onerror = (e) => {
      console.warn('Speech synthesis error:', e);
      this.isSpeaking = false;
      this.currentUtterance = null;
      if (this.onStateChange) this.onStateChange(false);
    };

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
    return true;
  }

  /**
   * Read out full disease advisory in structured speech
   */
  static speakDiseaseAdvisory(disease, language = 'en') {
    if (!disease) return;

    let speechText = '';
    if (language === 'hi') {
      speechText = `फसल रोग: ${disease.name}। ${disease.summary} मुख्य लक्षण: ${disease.symptoms.slice(0, 2).join(' ')}। जैविक उपचार: ${disease.treatment.organic[0]}। रासायनिक उपचार: ${disease.treatment.chemical[0]}। किसी भी रसायन के छिड़काव से पहले कृषि विशेषज्ञ की सलाह अवश्य लें।`;
    } else if (language === 'kn') {
      speechText = `ಬೆಳೆ ರೋಗ: ${disease.name}। ${disease.summary} ಪ್ರಮುಖ ಲಕ್ಷಣಗಳು: ${disease.symptoms.slice(0, 2).join(' ')}। ಸಾವಯವ ಚಿಕಿತ್ಸೆ: ${disease.treatment.organic[0]}। ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ: ${disease.treatment.chemical[0]}।`;
    } else {
      speechText = `Disease name: ${disease.name}. For ${disease.cropDisplayName}. Severity is ${disease.severity}. ${disease.summary}. Key visible signs: ${disease.visibleSigns}. Organic recommendation: ${disease.treatment.organic[0]}. Recommended chemical spray: ${disease.treatment.chemical[0]}. Important note: Consult your local agricultural officer before spraying.`;
    }

    return this.speak(speechText, language);
  }

  /**
   * Stop any active audio playback
   */
  static stop() {
    if (this.synth) {
      this.synth.cancel();
      this.isSpeaking = false;
      this.currentUtterance = null;
      if (this.onStateChange) this.onStateChange(false);
    }
  }

  /**
   * Subscribe to speaking state updates
   */
  static setListener(callback) {
    this.onStateChange = callback;
  }
}

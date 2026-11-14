/**
 * Offline Storage Service
 * Handles persistence for diagnosis history, user preferences, and offline form submissions
 */

import { INITIAL_SAMPLE_HISTORY } from '../data/sample_history.js';

const STORAGE_KEYS = {
  HISTORY: 'rootcause_diagnosis_history',
  SETTINGS: 'rootcause_user_settings',
  FORM_SUBMISSIONS: 'rootcause_offline_questions'
};

const DEFAULT_SETTINGS = {
  language: 'en',
  voiceEnabled: true,
  voiceSpeed: 1.0, // 0.85 for slow, 1.0 for normal
  offlineMode: true,
  apiEndpoint: '' // empty by default = offline engine
};

export class StorageService {
  /**
   * Get all diagnosis history records
   */
  static getHistory() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.HISTORY);
      if (!stored) {
        // Seed with initial realistic records on first launch
        localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(INITIAL_SAMPLE_HISTORY));
        return INITIAL_SAMPLE_HISTORY;
      }
      return JSON.parse(stored);
    } catch (e) {
      console.warn('LocalStorage error, falling back to memory:', e);
      return INITIAL_SAMPLE_HISTORY;
    }
  }

  /**
   * Save a new diagnosis record to local storage
   */
  static saveDiagnosis(record) {
    try {
      const history = this.getHistory();
      const newRecord = {
        id: 'diag_' + Date.now(),
        timestamp: new Date().toISOString(),
        displayDate: new Date().toLocaleDateString('en-GB', {
          day: '2-digit',
          month: 'short',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit'
        }),
        ...record
      };
      history.unshift(newRecord);
      localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(history));
      return newRecord;
    } catch (e) {
      console.error('Failed to save diagnosis:', e);
      return null;
    }
  }

  /**
   * Delete a diagnosis record by ID
   */
  static deleteDiagnosis(id) {
    try {
      const history = this.getHistory().filter(item => item.id !== id);
      localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify(history));
      return history;
    } catch (e) {
      console.error('Failed to delete diagnosis:', e);
      return [];
    }
  }

  /**
   * Clear all diagnosis records
   */
  static clearHistory() {
    try {
      localStorage.setItem(STORAGE_KEYS.HISTORY, JSON.stringify([]));
      return [];
    } catch (e) {
      console.error('Failed to clear history:', e);
      return [];
    }
  }

  /**
   * Get user settings
   */
  static getSettings() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      return stored ? { ...DEFAULT_SETTINGS, ...JSON.parse(stored) } : DEFAULT_SETTINGS;
    } catch (e) {
      return DEFAULT_SETTINGS;
    }
  }

  /**
   * Update user settings
   */
  static saveSettings(settings) {
    try {
      const current = this.getSettings();
      const updated = { ...current, ...settings };
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(updated));
      return updated;
    } catch (e) {
      console.error('Failed to save settings:', e);
      return DEFAULT_SETTINGS;
    }
  }

  /**
   * Save offline Jotform "Ask a Farming Question" submission
   */
  static saveQuestionSubmission(formData) {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.FORM_SUBMISSIONS);
      const list = stored ? JSON.parse(stored) : [];
      const entry = {
        id: 'q_' + Date.now(),
        timestamp: new Date().toISOString(),
        ...formData
      };
      list.unshift(entry);
      localStorage.setItem(STORAGE_KEYS.FORM_SUBMISSIONS, JSON.stringify(list));
      return entry;
    } catch (e) {
      console.error('Failed to save question:', e);
      return null;
    }
  }

  /**
   * Get stored questions
   */
  static getQuestionSubmissions() {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.FORM_SUBMISSIONS);
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      return [];
    }
  }
}

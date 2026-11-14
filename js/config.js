/**
 * RootCause API base URL.
 *
 * Resolution order:
 * 1. A base URL saved in Settings (localStorage key rootcause_user_settings.apiEndpoint)
 * 2. window.ROOTCAUSE_API_BASE, which can be edited in index.html before deployment
 * 3. The origin of this page when it is served over http or https
 *
 * An empty result means no model server can be reached from this page
 * (for example, the file was opened directly from disk).
 *
 * Use the computer's LAN address, such as http://192.168.1.20:8000, when
 * opening RootCause from a phone. localhost on the phone is the phone itself.
 */

function readSavedEndpoint() {
  try {
    const raw = localStorage.getItem('rootcause_user_settings');
    if (!raw) return '';
    const settings = JSON.parse(raw);
    return typeof settings.apiEndpoint === 'string' ? settings.apiEndpoint.trim() : '';
  } catch (err) {
    return '';
  }
}

export function normalizeApiBase(value) {
  if (!value) return '';
  let base = String(value).trim().replace(/\/+$/, '');
  base = base.replace(/\/predict$/i, '');
  base = base.replace(/\/api\/diagnose$/i, '');
  base = base.replace(/\/diagnose$/i, '');
  return base.replace(/\/+$/, '');
}

export function getApiBaseUrl() {
  const saved = normalizeApiBase(readSavedEndpoint());
  if (saved) return saved;

  const configured = (typeof window !== 'undefined' && window.ROOTCAUSE_API_BASE)
    ? normalizeApiBase(window.ROOTCAUSE_API_BASE)
    : '';
  if (configured) return configured;

  if (typeof window !== 'undefined' && window.location && /^https?:$/.test(window.location.protocol)) {
    return window.location.origin;
  }
  return '';
}

export function describeApiTarget() {
  const base = getApiBaseUrl();
  if (!base) {
    return 'No model server is configured. Open the app from the FastAPI server or set an API base URL in Settings.';
  }
  return base;
}

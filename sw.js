/**
 * RootCause Offline-First Service Worker
 * Pre-caches UI assets, CSS, images, and disease datasets
 */

const CACHE_NAME = 'rootcause-v1.1.0';
const ASSETS_TO_CACHE = [
  './',
  './index.html',
  './manifest.json',
  './css/main.css',
  './css/components.css',
  './css/animations.css',
  './js/app.js',
  './js/data/diseases.js',
  './js/data/translations.js',
  './js/data/sample_history.js',
  './js/services/storage.js',
  './js/services/classifier.js',
  './js/services/audio.js',
  './js/services/camera.js',
  './assets/icons/logo.svg',
  './assets/images/farmer_hero.svg',
  './assets/images/crop_rice.svg',
  './assets/images/crop_wheat.svg',
  './assets/images/sample_leaf_rice.svg',
  './assets/images/sample_leaf_wheat.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Pre-caching offline agricultural assets');
      return cache.addAll(ASSETS_TO_CACHE);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing obsolete cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  // Offline-first cache strategy
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).catch(() => {
        // Fallback for HTML documents if offline
        if (event.request.headers.get('accept')?.includes('text/html')) {
          return caches.match('./index.html');
        }
      });
    })
  );
});

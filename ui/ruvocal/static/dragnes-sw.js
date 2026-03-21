/**
 * DrAgnes Service Worker
 * Provides offline capability for dermoscopy analysis.
 *
 * Strategies:
 *  - Cache-first for WASM model weights and static assets
 *  - Network-first for brain API calls
 *  - Background sync for queued brain contributions
 */

const CACHE_VERSION = 'dragnes-v1';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const MODEL_CACHE = `${CACHE_VERSION}-model`;
const API_CACHE = `${CACHE_VERSION}-api`;

const STATIC_ASSETS = [
  '/dragnes',
  '/static/dragnes-manifest.json',
  '/static/dragnes-icon-192.svg',
  '/static/dragnes-icon-512.svg',
];

const MODEL_ASSETS = [
  '/static/wasm/rvagent_wasm.js',
  '/static/wasm/rvagent_wasm_bg.wasm',
];

// ---- Install ----------------------------------------------------------------

self.addEventListener('install', (event) => {
  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE).then((cache) => cache.addAll(STATIC_ASSETS)),
      caches.open(MODEL_CACHE).then((cache) => cache.addAll(MODEL_ASSETS)),
    ]).then(() => self.skipWaiting())
  );
});

// ---- Activate ---------------------------------------------------------------

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key.startsWith('dragnes-') && key !== STATIC_CACHE && key !== MODEL_CACHE && key !== API_CACHE)
          .map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

// ---- Fetch ------------------------------------------------------------------

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Network-first for brain API calls
  if (url.hostname === 'pi.ruv.io' || url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(event.request, API_CACHE));
    return;
  }

  // Cache-first for WASM model weights
  if (url.pathname.endsWith('.wasm') || url.pathname.includes('/wasm/')) {
    event.respondWith(cacheFirst(event.request, MODEL_CACHE));
    return;
  }

  // Cache-first for other static assets
  if (url.pathname.startsWith('/static/') || url.pathname.startsWith('/dragnes')) {
    event.respondWith(cacheFirst(event.request, STATIC_CACHE));
    return;
  }

  // Default: network only
  event.respondWith(fetch(event.request));
});

// ---- Background Sync --------------------------------------------------------

self.addEventListener('sync', (event) => {
  if (event.tag === 'dragnes-brain-sync') {
    event.waitUntil(syncBrainContributions());
  }
});

async function syncBrainContributions() {
  try {
    const cache = await caches.open(API_CACHE);
    const requests = await cache.keys();
    const pendingContributions = requests.filter((r) =>
      r.url.includes('brain') && r.method === 'POST'
    );

    for (const request of pendingContributions) {
      try {
        await fetch(request.clone());
        await cache.delete(request);
      } catch {
        // Will retry on next sync event
      }
    }
  } catch (error) {
    console.error('[DrAgnes SW] Background sync failed:', error);
  }
}

// ---- Push Notifications -----------------------------------------------------

self.addEventListener('push', (event) => {
  if (!event.data) return;

  const data = event.data.json();

  if (data.type === 'model-update') {
    event.waitUntil(
      Promise.all([
        self.registration.showNotification('DrAgnes Model Updated', {
          body: `Model ${data.version} is available with improved accuracy.`,
          icon: '/static/dragnes-icon-192.svg',
          badge: '/static/dragnes-icon-192.svg',
          tag: 'model-update',
        }),
        // Refresh cached model assets
        caches.open(MODEL_CACHE).then((cache) => cache.addAll(MODEL_ASSETS)),
      ])
    );
  }
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then((clients) => {
      const dragnesClient = clients.find((c) => c.url.includes('/dragnes'));
      if (dragnesClient) {
        return dragnesClient.focus();
      }
      return self.clients.openWindow('/dragnes');
    })
  );
});

// ---- Strategy helpers -------------------------------------------------------

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
  }
}

async function networkFirst(request, cacheName) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response(JSON.stringify({ error: 'offline' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

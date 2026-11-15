// Content-Security-Policy for the web UI (API security Step 4). Built per request by proxy.ts with a fresh nonce.
//
// Derived from what the UI actually loads:
// - scripts: only Next.js/React bundles and one inline theme script, all carrying the per-request nonce
//   ('strict-dynamic' lets the nonced bootstrap load its own chunks). No third-party scripts, no eval in production.
// - styles: the compiled Tailwind stylesheet (same origin) and nonced inline <style> from Next.js. React `style={...}`
//   attributes cannot carry a nonce, so only *attributes* are allowed inline (style-src-attr), not <style> elements.
// - API (separate origin, NEXT_PUBLIC_API_URL): fetch + the live event stream (connect-src), alert snapshots, camera
//   frames and MJPEG streams (img-src), processed videos (media-src).
// - blob: object URLs for local previews and downloads; data: for live frames (base64 JPEG).
// - nothing may frame the UI; no plugins; forms and <base> only to this origin.
// Development only adds what Next.js dev needs: 'unsafe-eval' (React dev tooling), inline styles (hot reload) and
// the hot-reload websocket. Not used: 'upgrade-insecure-requests' (the app is served over plain HTTP on this PC; add
// it together with HTTPS).
//
// Plain TypeScript with no imports, so it can be unit-tested with Node's built-in test runner.

export interface CspOptions {
  nonce: string
  apiOrigin: string          // e.g. http://localhost:8000
  dev: boolean
}

/** Origin (scheme://host:port) of the API base URL; falls back to the documented default on a bad value. */
export function apiOriginOf(apiBase: string | undefined): string {
  try {
    return new URL(apiBase || 'http://localhost:8000').origin
  } catch {
    return 'http://localhost:8000'
  }
}

export function buildCsp({ nonce, apiOrigin, dev }: CspOptions): string {
  const api = apiOrigin
  const directives: Record<string, string[]> = {
    'default-src': ["'self'"],
    'script-src': ["'self'", `'nonce-${nonce}'`, "'strict-dynamic'", ...(dev ? ["'unsafe-eval'"] : [])],
    'style-src': dev ? ["'self'", "'unsafe-inline'"] : ["'self'", `'nonce-${nonce}'`],
    'style-src-attr': ["'unsafe-inline'"],
    'img-src': ["'self'", 'data:', 'blob:', api],
    'media-src': ["'self'", 'blob:', api],
    'connect-src': ["'self'", api, ...(dev ? ['ws://localhost:*', 'ws://127.0.0.1:*'] : [])],
    'font-src': ["'self'", 'data:'],
    'object-src': ["'none'"],
    'base-uri': ["'self'"],
    'form-action': ["'self'"],
    'frame-ancestors': ["'none'"],
  }
  return Object.entries(directives).map(([k, v]) => `${k} ${v.join(' ')}`).join('; ')
}

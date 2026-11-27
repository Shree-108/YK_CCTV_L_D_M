// Content-Security-Policy for the web UI (API security Step 4). Built per request by proxy.ts with a fresh nonce.
//
// Derived from what the UI actually loads:
// - scripts: only Next.js/React bundles and one inline theme script, all carrying the per-request nonce
//   ('strict-dynamic' lets the nonced bootstrap load its own chunks). No third-party scripts, no eval in production.
// - styles: the compiled Tailwind stylesheet (same origin) and nonced inline <style> from Next.js. React `style={...}`
//   attributes cannot carry a nonce, so only *attributes* are allowed inline (style-src-attr), not <style> elements.
// - API: same origin (/ldm-api behind Nginx), so 'self' covers fetch + the live event stream, alert snapshots,
//   camera frames, MJPEG streams and processed videos.
// - blob: object URLs for local previews and downloads; data: for live frames (base64 JPEG).
// - nothing may frame the UI; no plugins; forms and <base> only to this origin.
// Development only adds what Next.js dev needs: 'unsafe-eval' (React dev tooling), inline styles (hot reload) and
// the hot-reload websocket. Not used: 'upgrade-insecure-requests' (the app may be served over plain HTTP; add it
// together with HTTPS).
//
// Plain TypeScript with no imports, so it can be unit-tested with Node's built-in test runner.

export interface CspOptions {
  nonce: string
  dev: boolean
}

export function buildCsp({ nonce, dev }: CspOptions): string {
  const directives: Record<string, string[]> = {
    'default-src': ["'self'"],
    'script-src': ["'self'", `'nonce-${nonce}'`, "'strict-dynamic'", ...(dev ? ["'unsafe-eval'"] : [])],
    'style-src': dev ? ["'self'", "'unsafe-inline'"] : ["'self'", `'nonce-${nonce}'`],
    'style-src-attr': ["'unsafe-inline'"],
    'img-src': ["'self'", 'data:', 'blob:'],
    'media-src': ["'self'", 'blob:'],
    'connect-src': ["'self'", ...(dev ? ['ws://localhost:*', 'ws://127.0.0.1:*'] : [])],
    'font-src': ["'self'", 'data:'],
    'object-src': ["'none'"],
    'base-uri': ["'self'"],
    'form-action': ["'self'"],
    'frame-ancestors': ["'none'"],
  }
  return Object.entries(directives).map(([k, v]) => `${k} ${v.join(' ')}`).join('; ')
}

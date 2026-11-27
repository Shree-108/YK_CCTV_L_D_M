/** @type {import('next').NextConfig} */
const nextConfig = {
  // deploy/start-ldm.ps1 builds the demo into its own folder so it never touches a running `next dev` (.next)
  distDir: process.env.NEXT_DIST_DIR || '.next',
  // Served at http://<host>/ldm behind Nginx (deploy/nginx/yk-cctv.conf). Keep equal to frontend/lib/paths.ts.
  basePath: '/ldm',
  // The browser calls the API same-origin at /ldm-api/... In production Nginx answers those requests before they
  // reach Next.js; this rewrite only serves them when Next.js is used directly (npm run dev, the Windows launchers).
  // LDM_API_ORIGIN is read on the server at build/start time only; it never reaches the browser.
  async rewrites() {
    const api = (process.env.LDM_API_ORIGIN || 'http://127.0.0.1:8000').replace(/\/$/, '')
    return [{ source: '/ldm-api/:path*', destination: `${api}/:path*`, basePath: false }]
  },
  typescript: {
    ignoreBuildErrors: false,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig

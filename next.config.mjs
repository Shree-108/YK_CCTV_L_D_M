/** @type {import('next').NextConfig} */
const nextConfig = {
  // deploy/start-ldm.ps1 builds the demo into its own folder so it never touches a running `next dev` (.next)
  distDir: process.env.NEXT_DIST_DIR || '.next',
  typescript: {
    ignoreBuildErrors: false,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig

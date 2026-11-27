// The dashboard is served under /ldm (next.config.mjs basePath; Nginx /ldm/ -> :3000) and the API under /ldm-api
// (Nginx /ldm-api/ -> :8000) on the same host. next/link and router.push add the base path themselves; plain
// <img src> and other raw URLs to files in public/ need asset().

export const BASE_PATH = '/ldm' // keep equal to basePath in next.config.mjs

/** URL of a file in public/, e.g. asset('/brand/logo.png') -> '/ldm/brand/logo.png'. */
export const asset = (path: string) => `${BASE_PATH}${path}`

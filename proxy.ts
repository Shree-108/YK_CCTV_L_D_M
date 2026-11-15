// Per-request Content-Security-Policy with a nonce (API security Step 4). Next.js reads the nonce from the request's
// CSP header and attaches it to its own scripts; the root layout passes it to the inline theme script. Policy:
// frontend/lib/csp.ts. Pages are therefore rendered per request (a nonce cannot be baked into a static page).
import { NextResponse, type NextRequest } from 'next/server'
import { apiOriginOf, buildCsp } from './frontend/lib/csp'

export function proxy(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString('base64')
  const csp = buildCsp({
    nonce,
    apiOrigin: apiOriginOf(process.env.NEXT_PUBLIC_API_URL),
    dev: process.env.NODE_ENV === 'development',
  })
  const requestHeaders = new Headers(request.headers)
  requestHeaders.set('x-nonce', nonce)
  requestHeaders.set('Content-Security-Policy', csp)
  const response = NextResponse.next({ request: { headers: requestHeaders } })
  response.headers.set('Content-Security-Policy', csp)
  response.headers.set('X-Content-Type-Options', 'nosniff')
  response.headers.set('Referrer-Policy', 'same-origin')
  return response
}

export const config = {
  matcher: [
    {
      // pages only: not static chunks, image optimisation, icons or brand files (they need no policy of their own)
      source: '/((?!_next/static|_next/image|favicon.ico|icon.svg|brand/).*)',
      missing: [
        { type: 'header', key: 'next-router-prefetch' },
        { type: 'header', key: 'purpose', value: 'prefetch' },
      ],
    },
  ],
}

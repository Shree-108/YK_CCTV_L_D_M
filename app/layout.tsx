import type { Metadata, Viewport } from 'next'
import { headers } from 'next/headers'
import './globals.css'
import { AppProvider } from '@/frontend/stores/app'
import { AuthGate } from '@/frontend/features/auth/AuthGate'
import { AppShell } from '@/frontend/components/shell/AppShell'
import { BRAND, MODULE_NAME } from '@/frontend/lib/module'

export const metadata: Metadata = {
  title: { default: `${BRAND} | ${MODULE_NAME}`, template: `%s · ${BRAND}` },
  description: 'CCTV litter detection: live monitoring, alert investigation, model evaluation and dataset management.',
  icons: { icon: [{ url: '/icon.svg', type: 'image/svg+xml' }] },
}

export const viewport: Viewport = {
  colorScheme: 'dark light',
  themeColor: '#0a0a0a',
}

// Applies the saved theme before first paint (prevents a light/dark flash).
const THEME_SCRIPT = `(function(){try{var t=JSON.parse(localStorage.getItem('lm.theme')||'"dark"');var d=t==='dark'||(t==='system'&&!matchMedia('(prefers-color-scheme: light)').matches);document.documentElement.dataset.theme=d?'dark':'light'}catch(e){document.documentElement.dataset.theme='dark'}})()`

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  // per-request nonce from proxy.ts (Content-Security-Policy): the only inline script must carry it
  const nonce = (await headers()).get('x-nonce') ?? undefined
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script nonce={nonce} dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        {/* Outermost: with LM_AUTH_MODE=on nothing else starts (streams, polling) until signed in; off = unchanged */}
        <AuthGate>
          <AppProvider>
            <AppShell>{children}</AppShell>
          </AppProvider>
        </AuthGate>
      </body>
    </html>
  )
}

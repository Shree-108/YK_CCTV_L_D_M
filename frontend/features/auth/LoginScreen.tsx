'use client'

// Sign-in screen (API security Step 3). The password is read from the form only at submit time and the field is
// cleared straight after - it is never kept in React state, storage or logs. Messages are fixed texts; raw server
// errors are never displayed.
import { useRef, useState, type FormEvent } from 'react'
import { AlertTriangle, Info, LogIn } from 'lucide-react'
import { login, type AuthMe } from '../../api/auth'
import { Button, Field, Input } from '../../components/ui/primitives'
import { BRAND, MODULE_NAME } from '../../lib/module'
import { asset } from '../../lib/paths'

export type LoginNotice = 'expired' | 'logged_out' | null

const NOTICE_TEXT: Record<Exclude<LoginNotice, null>, string> = {
  expired: 'Your session has expired. Please sign in again.',
  logged_out: 'You have been signed out.',
}

export function LoginScreen({ notice, onSignedIn, onModeOff }: {
  notice: LoginNotice
  onSignedIn: (me: AuthMe) => void
  onModeOff: () => void
}) {
  const formRef = useRef<HTMLFormElement>(null)
  const passwordRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (busy) return
    const data = new FormData(e.currentTarget)
    const username = String(data.get('username') ?? '').trim()
    const password = String(data.get('password') ?? '')
    if (!username || !password) {
      setError('Enter your username and password.')
      return
    }
    setBusy(true)
    setError(null)
    const res = await login(username, password)
    if (passwordRef.current) passwordRef.current.value = ''          // never keep the password around
    setBusy(false)
    if (res.ok) {
      onSignedIn(res.me)
      return
    }
    if (res.reason === 'disabled') {
      onModeOff()                                                   // login switched off on the server: re-check
      return
    }
    setError(res.message)
    passwordRef.current?.focus()
  }

  const info = notice ? NOTICE_TEXT[notice] : null
  return (
    <main className="flex min-h-dvh items-center justify-center bg-bg px-4 py-10">
      <div className="w-full max-w-[380px]">
        <div className="mb-6 flex flex-col items-center gap-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={asset('/brand/yatrikart-wordmark.png')} alt={BRAND} className="brand-logo h-[22px] w-auto select-none" draggable={false} />
          <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-fg-3">{MODULE_NAME}</div>
        </div>
        <section aria-labelledby="signin-title" className="rounded-lg border border-line bg-panel p-6 shadow-[var(--shadow)]">
          <h1 id="signin-title" className="text-[16px] font-semibold text-fg">Sign in</h1>
          <p className="mt-1 text-[12px] text-fg-3">Use the account created for you by the administrator.</p>

          {info && !error && (
            <div role="status" className="mt-4 flex items-start gap-2 rounded-md border border-line bg-panel-2 px-3 py-2 text-[12px] text-fg-2">
              <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden /> {info}
            </div>
          )}
          {error && (
            <div role="alert" className="mt-4 flex items-start gap-2 rounded-md bg-crit-bg px-3 py-2 text-[12px] text-crit">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden /> {error}
            </div>
          )}

          <form ref={formRef} onSubmit={submit} className="mt-4 flex flex-col gap-3" noValidate>
            <Field label="Username" htmlFor="login-username">
              <Input id="login-username" name="username" autoComplete="username" autoCapitalize="none" autoCorrect="off"
                spellCheck={false} maxLength={64} required autoFocus disabled={busy} />
            </Field>
            <Field label="Password" htmlFor="login-password">
              <Input id="login-password" name="password" type="password" autoComplete="current-password" maxLength={256}
                required ref={passwordRef} disabled={busy} />
            </Field>
            <Button type="submit" variant="primary" icon={LogIn} loading={busy} className="mt-1 h-9 w-full">
              {busy ? 'Signing in…' : 'Sign in'}
            </Button>
          </form>
        </section>
        <p className="mt-4 text-center text-[11px] text-fg-3">
          Forgot your password or locked out? Ask the administrator to reset or unlock your account.
        </p>
      </div>
    </main>
  )
}

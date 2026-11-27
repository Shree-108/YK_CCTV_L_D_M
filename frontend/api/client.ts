// Centralized API client. All data shown in the UI comes through here.

// Legacy admin token: never used while login is on, or before the server's mode is known (see adminToken.ts)
import { getAdminToken } from './adminToken'
export { getAdminToken, setAdminToken } from './adminToken'

// Same-origin API: Nginx routes /ldm-api/ to the FastAPI backend (next.config.mjs does the same for `next dev`).
export const API_BASE = '/ldm-api'

/** Fired on any 401 from the API; the login gate listens and re-checks the session. */
export const AUTH_REQUIRED_EVENT = 'ldm-auth-required'

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message)
  }
  get isNetwork() {
    return this.status === 0
  }
}

function messageFrom(detail: unknown, status: number): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (typeof d === 'object' && d && 'msg' in d ? `${(d as { loc?: unknown[] }).loc?.slice(-1)[0] ?? ''} ${(d as { msg: string }).msg}`.trim() : String(d)))
      .join('; ')
  }
  return `Request failed (${status})`
}

export interface RequestOptions {
  method?: string
  body?: unknown
  form?: FormData
  admin?: boolean
  signal?: AbortSignal
}

export async function api<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {}
  let body: BodyInit | undefined
  if (opts.form) body = opts.form
  else if (opts.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(opts.body)
  }
  const token = getAdminToken()
  if (token && (opts.admin || (opts.method && opts.method !== 'GET'))) headers.Authorization = `Bearer ${token}`
  let res: Response
  try {
    // credentials: the browser attaches the HttpOnly session cookie (it is never readable from JavaScript)
    res = await fetch(`${API_BASE}${path}`, { method: opts.method ?? (body ? 'POST' : 'GET'), headers, body, signal: opts.signal, credentials: 'include' })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError(0, `Backend not reachable at ${API_BASE}. Is the API server running?`)
  }
  if (!res.ok) {
    let detail: unknown
    try {
      detail = (await res.json()).detail
    } catch {
      detail = undefined
    }
    // 401 = no valid session (expired or revoked): let the login gate re-check right away instead of in 60 s
    if (res.status === 401 && typeof window !== 'undefined') window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT))
    throw new ApiError(res.status, messageFrom(detail, res.status), detail)
  }
  const ct = res.headers.get('content-type') ?? ''
  return (ct.includes('application/json') ? res.json() : res.text()) as Promise<T>
}

export const url = (path: string) => `${API_BASE}${path}`

/** Fixed, user-facing text for a failed download (server error bodies are never shown). */
export function downloadErrorMessage(status: number): string {
  if (status === 0) return `Backend not reachable at ${API_BASE}. Is the API server running?`
  if (status === 401) return 'Your session has ended. Please sign in again.'
  if (status === 403) return 'You do not have permission to download this. Ask an administrator.'
  if (status === 404) return 'Nothing to download: the file was not found.'
  return `The download failed (error ${status}). Please try again.`
}

/** Download a file through fetch (plain links cannot send the bearer token, and would show raw error JSON). */
export async function downloadAuthed(path: string, filename: string): Promise<void> {
  const token = getAdminToken()
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {}, credentials: 'include' })
  } catch {
    throw new ApiError(0, downloadErrorMessage(0))
  }
  if (!res.ok) {
    if (res.status === 401 && typeof window !== 'undefined') window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT))
    throw new ApiError(res.status, downloadErrorMessage(res.status))
  }
  const blob = await res.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ''
}

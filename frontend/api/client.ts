// Centralized API client. All data shown in the UI comes through here.

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

const TOKEN_KEY = 'lm.adminToken'

export function getAdminToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return window.sessionStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setAdminToken(token: string | null) {
  try {
    if (token) window.sessionStorage.setItem(TOKEN_KEY, token)
    else window.sessionStorage.removeItem(TOKEN_KEY)
    window.dispatchEvent(new Event('lm-admin-token'))
  } catch {
    /* storage unavailable: token lives only for this call */
  }
}

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
    res = await fetch(`${API_BASE}${path}`, { method: opts.method ?? (body ? 'POST' : 'GET'), headers, body, signal: opts.signal })
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
    throw new ApiError(res.status, messageFrom(detail, res.status), detail)
  }
  const ct = res.headers.get('content-type') ?? ''
  return (ct.includes('application/json') ? res.json() : res.text()) as Promise<T>
}

export const url = (path: string) => `${API_BASE}${path}`

/** Download a file from an admin-protected endpoint (plain links cannot send the bearer token). */
export async function downloadAuthed(path: string, filename: string): Promise<void> {
  const token = getAdminToken()
  const res = await fetch(`${API_BASE}${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
  if (!res.ok) {
    let msg = `Download failed (${res.status})`
    try { msg = (await res.json()).detail ?? msg } catch { /* not json */ }
    throw new ApiError(res.status, msg)
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

'use client'

import { useMemo } from 'react'
import { Api } from '../../api/endpoints'
import type { Camera, Store } from '../../api/types'
import { useQuery } from '../../hooks/useQuery'
import { camerasForStore, storeForCamera } from '../../lib/litter'
import { Field, Select } from '../../components/ui/primitives'

export interface StoreCamera { store_id: string; camera_id: string }

/** Registered stores (with their cameras) and cameras; shared cache keys so every page reuses one request. */
export function useStoresAndCameras() {
  const stores = useQuery('stores:list', Api.stores)
  const cams = useQuery('cameras:list', Api.cameras)
  return { stores: stores.data?.stores ?? [], cameras: cams.data?.cameras ?? [], error: stores.error ?? cams.error, loading: stores.loading || cams.loading }
}

/** Every camera known from either list (a store's camera may be missing from /api/cameras for a viewer). */
function allCameras(stores: Store[], cameras: Camera[]): { id: string; name: string; store_id: string | null }[] {
  const out = new Map(cameras.map((c) => [c.id, { id: c.id, name: c.name, store_id: c.store_id ?? null }]))
  for (const s of stores) for (const c of s.cameras) if (!out.has(c.id)) out.set(c.id, { id: c.id, name: c.name, store_id: s.id })
  return [...out.values()].sort((a, b) => a.id.localeCompare(b.id))
}

/**
 * Store + camera selectors. Choosing a registered camera fills in its store (the server rejects a detection attached
 * to a different store), so the store list is locked while a camera is selected. Empty = "Unknown": results are
 * recorded under UNKNOWN and the server returns a warning.
 */
export function StoreCameraSelect({ value, onChange, disabled, columns = 1 }: {
  value: StoreCamera; onChange: (v: StoreCamera) => void; disabled?: boolean; columns?: 1 | 2
}) {
  const { stores, cameras, error } = useStoresAndCameras()
  const cams = useMemo(() => allCameras(stores, cameras), [stores, cameras])
  const offered = camerasForStore(value.store_id, cams)
  const camStore = value.camera_id ? storeForCamera(value.camera_id, cams, stores) : ''
  const storeName = stores.find((s) => s.id === camStore)?.name

  const pickCamera = (camera_id: string) => onChange({ camera_id, store_id: camera_id ? storeForCamera(camera_id, cams, stores) : value.store_id })
  const pickStore = (store_id: string) => {
    const keep = value.camera_id && storeForCamera(value.camera_id, cams, stores) === store_id
    onChange({ store_id, camera_id: keep ? value.camera_id : '' })
  }

  return (
    <div className={columns === 2 ? 'grid gap-3 sm:grid-cols-2' : 'flex flex-col gap-3.5'}>
      <Field label="Store" hint={error ? 'Store list unavailable.' : value.camera_id
        ? (camStore ? `From camera ${value.camera_id}${storeName ? ` · ${storeName}` : ''}.` : `Camera ${value.camera_id} has no store: recorded under UNKNOWN.`)
        : value.store_id ? undefined : 'Not given: results are recorded under UNKNOWN.'}>
        <Select value={value.store_id} disabled={disabled || !!value.camera_id} onChange={(e) => pickStore(e.target.value)} aria-label="Store">
          <option value="">Unknown (not given)</option>
          {stores.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.name}</option>)}
          {value.store_id && !stores.some((s) => s.id === value.store_id) && <option value={value.store_id}>{value.store_id}</option>}
        </Select>
      </Field>
      <Field label="Camera" hint={value.camera_id ? undefined : 'Not given: results are recorded under UNKNOWN.'}>
        <Select value={value.camera_id} disabled={disabled} onChange={(e) => pickCamera(e.target.value)} aria-label="Camera">
          <option value="">Unknown (not given)</option>
          {offered.map((c) => <option key={c.id} value={c.id}>{c.id} · {c.name}</option>)}
          {value.camera_id && !offered.some((c) => c.id === value.camera_id) && <option value={value.camera_id}>{value.camera_id}</option>}
        </Select>
      </Field>
    </div>
  )
}

/** Adds store_id / camera_id to a multipart form (omitted when unknown: the server records UNKNOWN). */
export function appendIdentity(form: FormData, v: StoreCamera) {
  if (v.store_id) form.append('store_id', v.store_id)
  if (v.camera_id) form.append('camera_id', v.camera_id)
}

'use client'

// Monitored-region (automatic pavement ROI) summary for Image / Video Analysis results (both video pages) and the
// four-screenshot page. The polygon is drawn by the backend on annotated images / video, and by the browser as an
// SVG overlay (components/vision/Overlay.tsx, "Show monitored region" toggle). Colour / light-dark anomalies are not
// listed here or in the remark (phase 2: noise); they stay in the job JSON.
import Link from 'next/link'
import type { RoiBlock } from '../../api/types'
import { Badge } from '../../components/ui/primitives'

export const NO_LITTER_IN_REGION = 'No litter found in the monitored region'

export function RoiNotice({ roi, litter }: { roi?: RoiBlock | null; litter: number }) {
  if (!roi) return null
  const conf = typeof roi.confidence === 'number' ? ` · confidence ${roi.confidence.toFixed(2)}` : ''
  const cam = roi.camera_id && !['upload', 'UNKNOWN', ''].includes(roi.camera_id) ? roi.camera_id : null
  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-line bg-panel-2 px-3 py-2 text-[12px]" role="status">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-block size-2.5 rounded-sm" style={{ background: 'rgb(135 206 250)' }} aria-hidden />
        <span className="font-medium text-fg">Monitored region</span>
        {roi.applied
          ? <Badge tone="info">{roi.source}{conf}</Badge>
          : <Badge tone="warn">{roi.status === 'roi_pending' ? 'being computed' : 'not determined'}</Badge>}
        {roi.applied && (roi.needs_review
          ? <Badge tone="warn" title="Shown as needs_review in the remark until an operator approves the region">needs review · not approved</Badge>
          : <Badge tone="ok">approved</Badge>)}
        {roi.applied && cam && roi.needs_review && (
          <Link href={`/cameras?id=${encodeURIComponent(cam)}&tab=roi`} className="text-info underline-offset-2 hover:underline">Edit / Re-run / Approve on {cam}</Link>
        )}
        {roi.applied && !cam && roi.needs_review && (
          <span className="text-fg-3">Pick a camera on the upload form to save, edit and approve this region.</span>
        )}
      </div>
      {roi.warning && <p className="text-warn">{roi.warning}</p>}
      {roi.applied && litter === 0 && <p className="text-fg-2">{NO_LITTER_IN_REGION}{roi.message && roi.message !== NO_LITTER_IN_REGION ? ` (${roi.message})` : ''}.</p>}
      {(roi.notes ?? []).length > 0 && <p className="text-fg-3">{roi.notes!.join(' · ')}</p>}
    </div>
  )
}

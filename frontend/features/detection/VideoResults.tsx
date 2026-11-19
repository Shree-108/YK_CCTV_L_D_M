'use client'

// Results of a finished full-video analysis. "Unique litter events" (one per DETECTED track) and "frame
// observations" (every analysed frame the object was seen in) are different numbers and are never added together.
import Link from 'next/link'
import { Activity, FileDown, ListChecks, ScanSearch } from 'lucide-react'
import { url } from '../../api/client'
import type { Job, VideoLitterEvent, VideoResult } from '../../api/types'
import { className as cls, duration, num } from '../../lib/format'
import { Panel } from '../../components/ui/primitives'
import { BarList } from '../../components/ui/charts'
import { MetricCard } from '../../components/ui/kpi'
import { DataTable } from '../../components/ui/table'
import { EmptyState } from '../../components/ui/states'
import { ConfidenceCell, DetectionStatusBadge, Downloads, IdentityBlock, ProcessingBanner, Warnings } from './AnalysisOutputs'
import { litterName } from './common'

/** Download path of a finished job's file: the job's `files` map, then the analysis URLs. */
export function jobFile(job: Job<VideoResult>, kind: string): string | null {
  const f = job.files?.[kind]
  if (f) return f.startsWith('/') ? f : `/api/jobs/${encodeURIComponent(job.id)}/files/${kind}`
  return job.result?.analysis?.urls?.[kind] ?? (kind === 'video' && job.status === 'done' ? `/api/jobs/${encodeURIComponent(job.id)}/files/video` : null)
}

const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`
const shortTs = (s: string | null | undefined) => (s ? s.replace('T', ' ').slice(0, 19) : null)

function seek(jobId: string, t: number) {
  const v = document.querySelector<HTMLVideoElement>(`video[data-job="${jobId}"]`)
  if (v) { v.currentTime = Math.max(0, t - 0.5); void v.play().catch(() => {}); v.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }
}

export function VideoLitterResults({ job }: { job: Job<VideoResult> }) {
  const r = job.result ?? {}
  const lit = r.litter
  const vv = r.video_validation
  const validation = [
    ...(vv?.warnings ?? []),
    ...(vv?.truncated ? ['The video ended early (truncated): later frames could not be read.'] : []),
    ...(vv?.read_failures ? [`${num(vv.read_failures)} frame(s) could not be decoded.`] : []),
  ]
  const id = r.analysis?.analysis_id ?? job.id
  return (
    <>
      <Panel title="Litter results" icon={ScanSearch} subtitle={`${r.analysis?.analysis_id ?? job.id}${r.stats ? ` · model ${r.stats.model_version}` : ''}`}>
        <div className="flex flex-col gap-3">
          <ProcessingBanner status={r.analysis?.processing_status ?? (lit ? (lit.unique_events ? 'COMPLETED_LITTER_DETECTED' : 'COMPLETED_NO_LITTER_DETECTED') : null)} />
          <IdentityBlock identity={r.identity} />
          <Warnings title="Video validation" items={validation} />
          {!lit ? (
            <EmptyState compact title="No litter event data" body="This analysis was made before litter events were recorded. Analyse the video again to get unique events and frame observations." />
          ) : (
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
              <MetricCard label="Unique litter events" value={num(lit.unique_events)} tone={lit.unique_events ? 'crit' : undefined} hint="each object counted once" />
              <MetricCard label="Frame observations" value={num(lit.frame_observations)} hint="sightings across analysed frames" />
              <MetricCard label="Tracked objects" value={num(lit.tracked_objects)} hint="incl. tracks not confirmed" />
              <MetricCard label="Frames analysed" value={r.stats ? num(r.stats.frames_analysed) : '—'}
                hint={vv?.duration_sec != null ? `${duration(vv.duration_sec)} · ${vv.width}×${vv.height}${vv.fps ? ` · ${num(vv.fps, 1)} fps` : ''}` : undefined} />
            </div>
          )}
          <Downloads items={[
            { label: 'Annotated video', path: jobFile(job, 'video'), filename: `${id}_annotated.mp4` },
            { label: 'Detections CSV', path: jobFile(job, 'detections_csv'), filename: `${id}_detections.csv` },
            { label: 'Summary CSV', path: jobFile(job, 'summary_csv'), filename: `${id}_summary.csv` },
            { label: 'Metadata', path: jobFile(job, 'metadata'), filename: `${id}_metadata.json` },
            { label: 'Events JSON', path: jobFile(job, 'events'), filename: `${id}_events.json` },
          ]} />
        </div>
      </Panel>
      {lit && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <Panel title="By class" noPad>
            <DataTable rows={lit.by_class} rowKey={(c) => c.class} dense
              empty={<EmptyState compact title="No litter detected by the model" />}
              columns={[
                { key: 'cls', header: 'Class', cell: (c) => <span className="font-medium text-fg">{litterName(c.class)}</span>, sortValue: (c) => c.class },
                { key: 'ev', header: 'Unique events', align: 'right', cell: (c) => <span className="tabular text-fg">{num(c.events)}</span>, sortValue: (c) => c.events },
                { key: 'obs', header: 'Frame observations', align: 'right', cell: (c) => <span className="tabular text-fg-2">{num(c.observations)}</span>, sortValue: (c) => c.observations },
              ]} />
          </Panel>
          <EventsTable jobId={job.id} events={lit.events} />
        </div>
      )}
    </>
  )
}

function EventsTable({ jobId, events }: { jobId: string; events: VideoLitterEvent[] }) {
  return (
    <Panel title="Litter events" icon={ListChecks} subtitle="One row per unique object · click a row to jump to it in the video" noPad>
      <DataTable rows={events} rowKey={(e) => e.event_id} dense onRowClick={(e) => seek(jobId, e.first_seen_sec)}
        empty={<EmptyState compact title="No litter detected by the model" body="No track reached the DETECTED rule. This is not a guarantee that the area is litter-free." />}
        columns={[
          { key: 'img', header: 'Evidence', cell: (e) => e.evidence_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={url(e.evidence_url)} alt={`${litterName(e.class)} evidence, event ${e.event_id}`} loading="lazy" className="h-10 w-14 rounded border border-line bg-video object-cover" />
          ) : <span className="text-fg-3">—</span> },
          { key: 'id', header: 'Event', cell: (e) => <span className="font-mono text-[11.5px]">{e.event_id}</span>, sortValue: (e) => e.event_id },
          { key: 'cls', header: 'Class', cell: (e) => <span className="font-medium text-fg">{litterName(e.class)}</span>, sortValue: (e) => e.class },
          { key: 'first', header: 'First seen', sortValue: (e) => e.first_seen_sec, cell: (e) => (
            <span className="flex flex-col"><span className="font-mono text-info">{clock(e.first_seen_sec)}</span>{shortTs(e.first_seen) && <span className="text-[11px] text-fg-3">{shortTs(e.first_seen)}</span>}</span>) },
          { key: 'last', header: 'Last seen', sortValue: (e) => e.last_seen_sec, cell: (e) => (
            <span className="flex flex-col"><span className="font-mono">{clock(e.last_seen_sec)}</span>{shortTs(e.last_seen) && <span className="text-[11px] text-fg-3">{shortTs(e.last_seen)}</span>}</span>) },
          { key: 'dur', header: 'Duration', align: 'right', cell: (e) => duration(e.duration_sec), sortValue: (e) => e.duration_sec },
          { key: 'obs', header: 'Observations', align: 'right', cell: (e) => num(e.observation_count), sortValue: (e) => e.observation_count },
          { key: 'conf', header: 'Max confidence', cell: (e) => <ConfidenceCell value={e.max_confidence} />, sortValue: (e) => e.max_confidence },
          { key: 'st', header: 'Status', cell: (e) => <DetectionStatusBadge status={e.detection_status} /> },
        ]} />
    </Panel>
  )
}

/** Pipeline statistics (collapsed): model runs, timing, rejection reasons, raw JSON files. */
export function VideoTechnicalDetails({ job }: { job: Job<VideoResult> }) {
  const st = job.result!.stats!
  return (
    <details className="rounded-xl border border-line bg-panel">
      <summary className="cursor-pointer select-none px-4 py-3 text-[13px] font-medium text-fg-2 hover:text-fg">Technical details (pipeline statistics, raw files)</summary>
      <div className="grid gap-4 p-4 pt-0 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Panel title="Statistics" icon={Activity}
          subtitle={`${st.video.width}×${st.video.height} · ${num(st.video.fps, 1)} fps · ${duration(st.video.duration_sec)} · output ${st.video.output_codec}`}>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
            <MetricCard label="Frames analysed" value={num(st.frames_analysed)} hint={`of ${num(st.video.frames)} · every ${st.sample_every_n_frames} frame(s)`} />
            <MetricCard label="Model runs" value={num(st.model_runs ?? st.frames_analysed)} hint={st.frames_reused_static ? `${num(st.frames_reused_static)} static frames reused` : undefined} />
            <MetricCard label="Processing time" value={duration(st.wall_time_sec)} hint={st.device ? `on ${String(st.device).toUpperCase()}` : undefined} />
            <MetricCard label="Processing FPS" value={num(st.processing_fps, 1)} />
            <MetricCard label="Mean inference" value={st.mean_inference_ms != null ? `${num(st.mean_inference_ms, 0)} ms` : '—'} />
            <MetricCard label="Alerts created" value={num(st.new_alerts)} />
          </div>
          {st.new_alerts > 0 && <Link href="/alerts" className="mt-2 inline-block text-[12px] text-info hover:underline">Review {st.new_alerts} alert(s) →</Link>}
          <div className="mt-3 flex items-center gap-2 text-[12px] text-fg-3"><FileDown className="size-3.5" aria-hidden />Raw files</div>
          <div className="mt-1.5">
            <Downloads items={(['detections', 'stats'] as const).map((k) => ({ label: `${k}.json`, path: jobFile(job, k), filename: `${job.id}_${k}.json` }))} />
          </div>
        </Panel>
        <Panel title="Detections by status">
          <BarList data={Object.entries(st.detections_by_status).map(([k, v]) => ({ label: k.replace(/_/g, ' '), value: v }))} />
          {Object.keys(st.rejection_reasons).length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-fg-3">Rejection reasons</h3>
              <BarList color="var(--fg-3)" data={Object.entries(st.rejection_reasons).map(([k, v]) => ({ label: cls(k), value: v }))} />
            </div>
          )}
        </Panel>
      </div>
    </details>
  )
}

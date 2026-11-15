'use client'

// Static help content. Frontend-only: it only links to existing pages and makes no API calls.
import Link from 'next/link'
import {
  AlertTriangle, ArrowRight, BarChart3, Camera, Cctv, FileVideo, ImageIcon, Keyboard, LayoutGrid, Settings, type LucideIcon,
} from 'lucide-react'
import { Kbd, PageHeader, Panel } from '../../components/ui/primitives'

const GUIDES: { href: string; title: string; icon: LucideIcon; steps: string[] }[] = [
  { href: '/detect/image', title: 'Analyse an image', icon: ImageIcon, steps: [
    'Open Image Analysis and drop a CCTV still (JPEG, PNG, WebP or BMP).',
    'Optionally define the ROI: click the corners of the outdoor ground area, then Save ROI.',
    'Choose detection settings and press Analyze Image.',
    'Review the annotated preview and the detection table (category, confidence, priority).',
  ] },
  { href: '/detect/video', title: 'Analyse a video', icon: FileVideo, steps: [
    'Open Video Analysis and drop a CCTV recording (MP4, MOV, AVI, MKV, M4V or WebM).',
    'Draw the ROI on the first frame if litter should only be counted in one area.',
    'Adjust sampling and temporal confirmation, then press Analyze Video.',
    'When processing finishes, the annotated video plays; click timestamps to jump to each detection.',
  ] },
  { href: '/live-cctv', title: 'Monitor a live camera', icon: Cctv, steps: [
    'Add the camera under Camera Integration (RTSP / NVR details).',
    'Open Live CCTV, select the camera and connect.',
    'Confirmed litter inside the ROI raises an alert automatically.',
  ] },
  { href: '/alerts', title: 'Work through alerts', icon: AlertTriangle, steps: [
    'New alerts appear under the New tab, highest priority first when sorted.',
    'Open an alert to see the evidence snapshot, crop and observation timeline.',
    'Acknowledge, Resolve or Reject (false positive); add operator notes if needed.',
    'Export the filtered list with Export CSV (administrators only when sign-in is enabled).',
  ] },
]

const PAGES: { href: string; label: string; icon: LucideIcon; body: string }[] = [
  { href: '/', label: 'Overview', icon: LayoutGrid, body: 'Totals, priority incidents, trends and the latest evidence.' },
  { href: '/analytics', label: 'Analytics', icon: BarChart3, body: 'Trends by day, hour, category and camera with filters.' },
  { href: '/cameras', label: 'Camera Integration', icon: Camera, body: 'Camera list, RTSP configuration and per-camera ROI.' },
  { href: '/settings', label: 'Settings', icon: Settings, body: 'Thresholds, alert rules, appearance and security.' },
]

const STATUSES: [string, string, string][] = [
  ['bg-ok', 'Green', 'Healthy, connected or resolved.'],
  ['bg-warn', 'Amber', 'Needs attention: degraded, pending review or medium priority.'],
  ['bg-crit', 'Red', 'Critical: new alert, high priority, offline or failed.'],
]

export default function HelpCenterPage() {
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Help Center" description="How to run litter detection on images, videos and live CCTV, and how to act on the results." />

      <div className="grid gap-4 lg:grid-cols-2">
        {GUIDES.map((g) => (
          <Panel key={g.href} title={g.title} icon={g.icon}
            actions={<Link href={g.href} className="inline-flex items-center gap-1 text-[12px] font-medium text-fg-2 hover:text-fg">Open <ArrowRight className="size-3.5" aria-hidden /></Link>}>
            <ol className="flex flex-col gap-2.5">
              {g.steps.map((s, i) => (
                <li key={i} className="flex gap-3 text-[13px] text-fg-2">
                  <span className="flex size-5 shrink-0 items-center justify-center rounded-full border border-line-strong text-[11px] font-semibold text-fg">{i + 1}</span>
                  <span className="pt-px">{s}</span>
                </li>
              ))}
            </ol>
          </Panel>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <Panel title="Other pages">
          <ul className="grid gap-2 sm:grid-cols-2">
            {PAGES.map((p) => (
              <li key={p.href}>
                <Link href={p.href} className="flex h-full gap-3 rounded-lg border border-line bg-panel-2 p-3 transition-colors hover:border-line-strong hover:bg-hover">
                  <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-info-bg text-fg"><p.icon className="size-4" aria-hidden /></span>
                  <span className="min-w-0">
                    <span className="block text-[13px] font-medium text-fg">{p.label}</span>
                    <span className="block text-[12px] text-fg-3">{p.body}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </Panel>
        <div className="flex flex-col gap-4">
          <Panel title="Status colours">
            <ul className="flex flex-col gap-2">
              {STATUSES.map(([dot, name, body]) => (
                <li key={name} className="flex items-start gap-2.5 text-[13px]">
                  <span className={`mt-1.5 size-2 shrink-0 rounded-full ${dot}`} aria-hidden />
                  <span><span className="font-medium text-fg">{name}</span> <span className="text-fg-3">— {body}</span></span>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Keyboard" icon={Keyboard}>
            <div className="flex items-center justify-between text-[13px] text-fg-2">
              Search and jump to any page
              <span className="flex gap-0.5"><Kbd>Ctrl</Kbd><Kbd>K</Kbd></span>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  )
}

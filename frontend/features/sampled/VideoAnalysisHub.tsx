'use client'

// Video Analysis: the four-screenshot "Quick check" is the default; the full-video tracking analysis stays available
// as a secondary mode (?mode=full). The mode lives only in the URL, so refresh/remount/back keep it.
import Link from 'next/link'
import { usePathname, useRouter, useSearchParams } from 'next/navigation'
import { BarChart3, Film, Images } from 'lucide-react'
import { PageHeader, Segmented } from '../../components/ui/primitives'
import VideoDetectionPage from '../detection/VideoDetectionPage'
import SampledAnalysisPage from './SampledAnalysisPage'

export default function VideoAnalysisHub() {
  const params = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  // full-video tracking only when asked for (?mode=full) or when opening an earlier full-video job (?video_job=,
  // links from the Overview); everything else - including no parameter - opens the quick check
  const m = params.get('mode')
  const mode = m === 'full' || (!m && params.has('video_job')) ? 'full' : 'quick'
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Video Analysis"
        description={mode === 'full'
          ? 'Every sampled frame of the video is analysed and litter is tracked: each object is one litter event. The annotated video plays when the analysis is complete.'
          : 'Quick check: four screenshots of the video are analysed; annotated images and CSV reports are produced automatically.'}
        actions={<>
          <Segmented label="Analysis type" value={mode} onChange={(v) => router.replace(v === 'full' ? `${pathname}?mode=full` : pathname, { scroll: false })}
            options={[{ value: 'quick', label: 'Quick check (4 screenshots)', icon: Images }, { value: 'full', label: 'Full video (tracking)', icon: Film }]} />
          <Link href="/results" className="inline-flex h-8 items-center gap-1.5 rounded-md border border-line bg-panel-2 px-3 text-[13px] font-medium text-fg hover:border-line-strong hover:bg-hover">
            <BarChart3 className="size-3.5" aria-hidden />Quick-check results
          </Link>
        </>} />
      {mode === 'full' ? <VideoDetectionPage embedded /> : <SampledAnalysisPage />}
    </div>
  )
}

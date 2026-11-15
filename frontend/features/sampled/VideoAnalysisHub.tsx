'use client'

// Video Analysis: the four-screenshot workflow is the default; the earlier full-video tracking analysis stays
// available unchanged under "Full video (tracking)".
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
  const mode = params.get('mode') === 'full' ? 'full' : 'sampled'
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Video Analysis"
        description={mode === 'sampled'
          ? 'Drop a video: four screenshots, litter detection, annotated images and a CSV report are produced automatically.'
          : 'Every sampled frame of the video is tracked; the processed video plays when the analysis is complete.'}
        actions={<>
          <Segmented label="Analysis type" value={mode} onChange={(v) => router.replace(v === 'full' ? `${pathname}?mode=full` : pathname, { scroll: false })}
            options={[{ value: 'sampled', label: 'Four screenshots', icon: Images }, { value: 'full', label: 'Full video (tracking)', icon: Film }]} />
          <Link href="/results" className="inline-flex h-8 items-center gap-1.5 rounded-md border border-line bg-panel-2 px-3 text-[13px] font-medium text-fg hover:border-line-strong hover:bg-hover">
            <BarChart3 className="size-3.5" aria-hidden />Results Dashboard
          </Link>
        </>} />
      {mode === 'sampled' ? <SampledAnalysisPage /> : <VideoDetectionPage embedded />}
    </div>
  )
}

'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { FileVideo, ImageIcon } from 'lucide-react'
import { Segmented } from '../../components/ui/primitives'
import ImageDetectionPage from './ImageDetectionPage'
import VideoDetectionPage from './VideoDetectionPage'

/** Single upload workspace: image or video analysis. */
export default function UploadPage() {
  const params = useSearchParams()
  const router = useRouter()
  const mode = params.get('mode') === 'video' ? 'video' : 'image'
  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <Segmented label="Upload type" value={mode} onChange={(v) => router.replace(`/upload?mode=${v}`)}
          options={[{ value: 'image', label: 'Image', icon: ImageIcon }, { value: 'video', label: 'Video', icon: FileVideo }]} />
      </div>
      {mode === 'image' ? <ImageDetectionPage /> : <VideoDetectionPage />}
    </div>
  )
}

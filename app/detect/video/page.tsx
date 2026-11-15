import { Suspense } from 'react'
import Feature from '@/frontend/features/sampled/VideoAnalysisHub'

export const metadata = { title: 'Video Analysis' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

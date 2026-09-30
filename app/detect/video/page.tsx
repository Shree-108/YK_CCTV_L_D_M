import { Suspense } from 'react'
import Feature from '@/frontend/features/detection/VideoDetectionPage'

export const metadata = { title: 'Video Detection' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

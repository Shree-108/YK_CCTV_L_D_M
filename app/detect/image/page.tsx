import { Suspense } from 'react'
import Feature from '@/frontend/features/detection/ImageDetectionPage'

export const metadata = { title: 'Image Detection' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

import { Suspense } from 'react'
import Feature from '@/frontend/features/labeling/LabelingPage'

export const metadata = { title: 'Label mined detections' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

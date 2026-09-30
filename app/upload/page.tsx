import { Suspense } from 'react'
import Feature from '@/frontend/features/detection/UploadPage'

export const metadata = { title: 'Upload analysis' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

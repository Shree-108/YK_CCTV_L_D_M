import { Suspense } from 'react'
import Feature from '@/frontend/features/cameras/LiveCctvPage'

export const metadata = { title: 'Live CCTV' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

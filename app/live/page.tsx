import { Suspense } from 'react'
import Feature from '@/frontend/features/cameras/LivePage'

export const metadata = { title: 'Live Cameras' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

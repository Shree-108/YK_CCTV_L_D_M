import { Suspense } from 'react'
import Feature from '@/frontend/features/cameras/CamerasPage'

export const metadata = { title: 'Cameras' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

import { Suspense } from 'react'
import Feature from '@/frontend/features/data/GroundTruthPage'

export const metadata = { title: 'Ground Truth' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

import { Suspense } from 'react'
import Feature from '@/frontend/features/results/ChecksPage'

export const metadata = { title: 'Results' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

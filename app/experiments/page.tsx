import { Suspense } from 'react'
import Feature from '@/frontend/features/models/ExperimentsPage'

export const metadata = { title: 'Experiments' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

import { Suspense } from 'react'
import Feature from '@/frontend/features/data/HardExamplesPage'

export const metadata = { title: 'Hard Examples' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

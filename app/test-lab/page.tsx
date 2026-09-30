import { Suspense } from 'react'
import Feature from '@/frontend/features/models/TestLabPage'

export const metadata = { title: 'Test Lab' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

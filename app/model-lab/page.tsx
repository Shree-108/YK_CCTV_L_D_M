import { Suspense } from 'react'
import Feature from '@/frontend/features/models/ModelLabPage'

export const metadata = { title: 'Model Lab' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

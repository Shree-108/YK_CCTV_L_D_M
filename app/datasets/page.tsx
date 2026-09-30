import { Suspense } from 'react'
import Feature from '@/frontend/features/data/DatasetsPage'

export const metadata = { title: 'Dataset Lab' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

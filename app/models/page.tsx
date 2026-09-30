import { Suspense } from 'react'
import Feature from '@/frontend/features/models/ModelsPage'

export const metadata = { title: 'Models' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

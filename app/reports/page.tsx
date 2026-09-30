import { Suspense } from 'react'
import Feature from '@/frontend/features/analytics/ReportsPage'

export const metadata = { title: 'Reports' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

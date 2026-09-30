import { Suspense } from 'react'
import Feature from '@/frontend/features/dashboard/DashboardPage'

export const metadata = { title: 'Dashboard' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

import { Suspense } from 'react'
import Feature from '@/frontend/features/dashboard/DashboardPage'

export const metadata = { title: 'Dashboard · Yatrikart' }   // the layout's title template does not apply to the root page itself

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

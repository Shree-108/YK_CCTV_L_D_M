import { Suspense } from 'react'
import Feature from '@/frontend/features/alerts/HistoryPage'

export const metadata = { title: 'History' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

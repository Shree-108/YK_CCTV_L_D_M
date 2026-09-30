import { Suspense } from 'react'
import Feature from '@/frontend/features/system/SystemHealthPage'

export const metadata = { title: 'System Health' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

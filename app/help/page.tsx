import { Suspense } from 'react'
import Feature from '@/frontend/features/system/HelpCenterPage'

export const metadata = { title: 'Help Center' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

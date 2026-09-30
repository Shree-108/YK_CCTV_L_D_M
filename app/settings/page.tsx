import { Suspense } from 'react'
import Feature from '@/frontend/features/system/SettingsPage'

export const metadata = { title: 'Settings' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

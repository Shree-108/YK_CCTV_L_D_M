import { Suspense } from 'react'
import Feature from '@/frontend/features/models/EvaluationPage'

export const metadata = { title: 'Evaluation' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

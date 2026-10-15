import { Suspense } from 'react'
import Feature from '@/frontend/features/invoices/InvoiceExtractionPage'

export const metadata = { title: 'Invoice Extraction' }

export default function Page() {
  return (
    <Suspense>
      <Feature />
    </Suspense>
  )
}

'use client'

import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { cx } from '../../lib/format'
import { Sparkline } from './charts'
import { Skeleton } from './states'
import type { Tone } from './primitives'

const TONE_TEXT: Record<Tone, string> = { ok: 'text-ok', warn: 'text-warn', crit: 'text-crit', info: 'text-info', exp: 'text-exp', neutral: 'text-fg-3' }
const TONE_CHIP: Record<Tone, string> = { ok: 'bg-ok-bg text-ok', warn: 'bg-warn-bg text-warn', crit: 'bg-crit-bg text-crit', info: 'bg-info-bg text-fg', exp: 'bg-exp-bg text-exp', neutral: 'bg-panel-2 text-fg-2' }

/**
 * KPI tile. `value` of null renders an em dash (no data) — never a made-up number.
 */
export function KpiCard({ label, value, sub, icon: Icon, tone = 'neutral', spark, loading, onClick, status }: {
  label: string; value: ReactNode | null; sub?: ReactNode; icon?: LucideIcon; tone?: Tone; spark?: number[]; loading?: boolean; onClick?: () => void; status?: ReactNode
}) {
  const Comp = onClick ? 'button' : 'div'
  return (
    <Comp onClick={onClick}
      className={cx('flex min-w-0 flex-col gap-1.5 rounded-xl border border-line bg-panel p-4 text-left', onClick && 'transition-colors hover:border-line-strong hover:bg-panel-2')}>
      <div className="flex items-start justify-between gap-2">
        <span className="truncate pt-0.5 text-[12px] font-medium text-fg-3">{label}</span>
        {Icon && <span className={cx('flex size-8 shrink-0 items-center justify-center rounded-lg', TONE_CHIP[tone])}><Icon className="size-4" aria-hidden /></span>}
      </div>
      {loading ? <Skeleton className="h-8 w-20" /> : (
        <div className="tabular truncate text-[26px] font-semibold leading-8 tracking-tight text-fg">{value ?? '—'}</div>
      )}
      <div className="flex min-h-4 items-center justify-between gap-2">
        <span className="truncate text-[11px] text-fg-3">{sub}</span>
        {status}
      </div>
      {spark && spark.length > 1 && <Sparkline values={spark} color={tone === 'neutral' ? 'var(--info)' : `var(--${tone})`} className="mt-0.5" />}
    </Comp>
  )
}

export function MetricCard({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: Tone }) {
  return (
    <div className="min-w-0 rounded-lg border border-line bg-panel-2 px-3.5 py-2.5">
      <div className="truncate text-[11.5px] font-medium text-fg-3">{label}</div>
      <div className={cx('tabular truncate text-[19px] font-semibold leading-7', tone ? TONE_TEXT[tone] : 'text-fg')}>{value}</div>
      {hint && <div className="truncate text-[11px] text-fg-3">{hint}</div>}
    </div>
  )
}

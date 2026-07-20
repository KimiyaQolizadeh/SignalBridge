import type { ReactNode } from 'react'

interface MetricCardProps {
  label: string
  value: ReactNode
  detail?: string
  tone?: 'neutral' | 'positive' | 'warning'
}

export function MetricCard({
  label,
  value,
  detail,
  tone = 'neutral',
}: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
      {detail ? <span className="metric-card__detail">{detail}</span> : null}
    </article>
  )
}

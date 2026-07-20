import { getStatusPresentation } from '../ui/status'

interface StatusBadgeProps {
  value: string | null | undefined
  label?: string
}

export function StatusBadge({ value, label }: StatusBadgeProps) {
  const presentation = getStatusPresentation(value)
  return (
    <span className={`status-badge status-badge--${presentation.tone}`}>
      {label ?? presentation.label}
    </span>
  )
}

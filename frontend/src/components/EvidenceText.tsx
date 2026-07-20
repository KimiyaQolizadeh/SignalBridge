interface EvidenceTextProps {
  text: string
  quote?: string | null
}

export function EvidenceText({ text, quote }: EvidenceTextProps) {
  if (!quote) return <>{text}</>
  const exactIndex = text.indexOf(quote)
  const index = exactIndex >= 0 ? exactIndex : text.toLocaleLowerCase().indexOf(quote.toLocaleLowerCase())
  if (index < 0) return <>{text}</>
  return (
    <>
      {text.slice(0, index)}
      <mark className="evidence-highlight">{text.slice(index, index + quote.length)}</mark>
      {text.slice(index + quote.length)}
    </>
  )
}

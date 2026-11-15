/** Minimal RFC 4180 CSV parser (quoted fields, escaped quotes, CRLF) for previewing server-generated reports. */
export function parseCsv(text: string): { header: string[]; rows: Record<string, string>[] } {
  const out: string[][] = []
  let row: string[] = [], field = '', q = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (c === '"') q = false
      else field += c
    } else if (c === '"') q = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field); field = ''
      if (row.length > 1 || row[0] !== '') out.push(row)
      row = []
    } else field += c
  }
  if (field !== '' || row.length) { row.push(field); out.push(row) }
  const [header = [], ...body] = out
  return { header, rows: body.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? '']))) }
}

// Unit tests: CSV preview parser (frontend/lib/csv.ts). Run: npm run test:frontend
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { parseCsv } from '../lib/csv.ts'

test('header and rows, including the header-only report', () => {
  assert.deepEqual(parseCsv('a,b\r\n1,2\r\n3,\r\n'), { header: ['a', 'b'], rows: [{ a: '1', b: '2' }, { a: '3', b: '' }] })
  assert.deepEqual(parseCsv('video_name,store_id\n'), { header: ['video_name', 'store_id'], rows: [] })
})

test('quoted fields keep commas, quotes and line breaks', () => {
  const { rows } = parseCsv('remark,n\n"No litter, near ""the"" door",1\n"two\nlines",2')
  assert.equal(rows[0].remark, 'No litter, near "the" door')
  assert.equal(rows[1].remark, 'two\nlines')
  assert.equal(rows[1].n, '2')
})

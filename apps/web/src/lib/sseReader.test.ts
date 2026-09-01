import { describe, expect, it, vi } from 'vitest'

import { readSSEStream } from './sseReader'

function makeSSEResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  let index = 0
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]))
        index++
      } else {
        controller.close()
      }
    },
  })
  return { body: stream } as unknown as Response
}

describe('readSSEStream', () => {
  it('throws when response body is missing', async () => {
    const response = { body: null } as unknown as Response
    await expect(readSSEStream(response, vi.fn())).rejects.toThrow('Streaming not supported')
  })

  it('parses simple SSE events', async () => {
    const response = makeSSEResponse([
      'event: step\ndata: hello\n\n',
    ])
    const events: Array<{ type: string; data: string }> = []
    await readSSEStream(response, (type, data) => {
      events.push({ type, data })
    })
    expect(events).toEqual([{ type: 'step', data: 'hello' }])
  })

  it('parses multiple events in a single chunk', async () => {
    const response = makeSSEResponse([
      'event: a\ndata: first\n\nevent: b\ndata: second\n\n',
    ])
    const events: Array<{ type: string; data: string }> = []
    await readSSEStream(response, (type, data) => {
      events.push({ type, data })
    })
    expect(events).toEqual([
      { type: 'a', data: 'first' },
      { type: 'b', data: 'second' },
    ])
  })

  it('handles events split across chunks', async () => {
    const response = makeSSEResponse([
      'event: step\n',
      'data: split\n\n',
    ])
    const events: Array<{ type: string; data: string }> = []
    await readSSEStream(response, (type, data) => {
      events.push({ type, data })
    })
    expect(events).toEqual([{ type: 'step', data: 'split' }])
  })

  it('handles event and data in same chunk but split by newline', async () => {
    const response = makeSSEResponse([
      'event: step\ndata: together\n',
    ])
    const events: Array<{ type: string; data: string }> = []
    await readSSEStream(response, (type, data) => {
      events.push({ type, data })
    })
    expect(events).toEqual([{ type: 'step', data: 'together' }])
  })

  it('handles data lines without a preceding event line', async () => {
    const response = makeSSEResponse([
      'data: no-event\n',
    ])
    const events: Array<{ type: string; data: string }> = []
    await readSSEStream(response, (type, data) => {
      events.push({ type, data })
    })
    expect(events).toEqual([{ type: '', data: 'no-event' }])
  })

  it('ignores non-event and non-data lines', async () => {
    const response = makeSSEResponse([
      'comment: ignored\nevent: real\ndata: value\n',
    ])
    const events: Array<{ type: string; data: string }> = []
    await readSSEStream(response, (type, data) => {
      events.push({ type, data })
    })
    expect(events).toEqual([{ type: 'real', data: 'value' }])
  })

  it('does not emit an event id twice when a reconnect reuses the seen-id set', async () => {
    const seen = new Set<string>()
    const events: string[] = []
    await readSSEStream(makeSSEResponse(['id: 42\nevent: content\ndata: first\n\n']), (_type, data) => events.push(data), seen)
    await readSSEStream(makeSSEResponse(['id: 42\nevent: content\ndata: duplicate\n\nid: 43\nevent: done\ndata: {}\n\n']), (_type, data) => events.push(data), seen)
    expect(events).toEqual(['first', '{}'])
  })
})

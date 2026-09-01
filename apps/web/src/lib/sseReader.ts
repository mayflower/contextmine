/**
 * Read an SSE (Server-Sent Events) stream from a fetch Response,
 * calling `onEvent` for each parsed event/data pair.
 */
export async function readSSEStream(
  response: Response,
  onEvent: (eventType: string, data: string) => void,
  seenEventIds: Set<string> = new Set(),
): Promise<void> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('Streaming not supported')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  const dispatch = (frame: string) => {
    let eventType = ''
    let eventId = ''
    const data: string[] = []
    for (const rawLine of frame.split(/\r?\n/)) {
      if (rawLine.startsWith('event:')) eventType = rawLine.slice(6).trimStart()
      else if (rawLine.startsWith('id:')) eventId = rawLine.slice(3).trimStart()
      else if (rawLine.startsWith('data:')) data.push(rawLine.slice(5).trimStart())
    }
    if (data.length === 0 || (eventId && seenEventIds.has(eventId))) return
    if (eventId) seenEventIds.add(eventId)
    onEvent(eventType, data.join('\n'))
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      buffer += decoder.decode()
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split(/\r?\n\r?\n/)
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      dispatch(frame)
    }
  }

  if (buffer.trim()) dispatch(buffer)
}

/**
 * Read an SSE (Server-Sent Events) stream from a fetch Response,
 * calling `onEvent` for each parsed event/data pair.
 */
export async function readSSEStream(
  response: Response,
  onEvent: (eventType: string, data: string) => void,
): Promise<void> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('Streaming not supported')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE events from buffer
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // Keep incomplete line in buffer

    let eventType = ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7)
      } else if (line.startsWith('data: ')) {
        const data = line.slice(6)
        onEvent(eventType, data)
      }
    }
  }
}

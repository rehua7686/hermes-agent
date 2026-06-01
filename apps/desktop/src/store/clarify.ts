import { atom } from 'nanostores'

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
}

export interface ClarifyInputState {
  draft: string
  focusLocked: boolean
  scrollTop: number
  selectionEnd: number | null
  selectionStart: number | null
  typing: boolean
}

export interface ClarifyTextareaPosition {
  scrollTop: number
  selectionEnd: number
  selectionStart: number
}

// Holds the request_id (and metadata) for the most recent in-flight
// clarify call. The inline ClarifyTool component (rendered inside the
// assistant message stream) reads this to know which request_id to send
// back over `clarify.respond`.
export const $clarifyRequest = atom<ClarifyRequest | null>(null)

export const $clarifyInputs = atom<Record<string, ClarifyInputState>>({})

function normalizeClarifyInput(input?: Partial<ClarifyInputState>): ClarifyInputState {
  return {
    draft: input?.draft ?? '',
    focusLocked: input?.focusLocked ?? false,
    scrollTop: input?.scrollTop ?? 0,
    selectionEnd: input?.selectionEnd ?? null,
    selectionStart: input?.selectionStart ?? null,
    typing: input?.typing ?? false
  }
}

function updateClarifyInput(key: string, patch: Partial<ClarifyInputState>): void {
  const current = $clarifyInputs.get()
  const previous = normalizeClarifyInput(current[key])
  const next = { ...previous, ...patch }

  if (
    previous.draft === next.draft &&
    previous.focusLocked === next.focusLocked &&
    previous.scrollTop === next.scrollTop &&
    previous.selectionEnd === next.selectionEnd &&
    previous.selectionStart === next.selectionStart &&
    previous.typing === next.typing
  ) {
    return
  }

  $clarifyInputs.set({ ...current, [key]: next })
}

export function clarifyInputKey(requestId?: null | string, question?: string): string {
  const id = requestId?.trim()

  if (id) {
    return `request:${id}`
  }

  const normalizedQuestion = question?.trim()

  return normalizedQuestion ? `question:${normalizedQuestion}` : 'pending'
}

export function setClarifyRequest(request: ClarifyRequest): void {
  const idKey = clarifyInputKey(request.requestId, request.question)
  const questionKey = clarifyInputKey(null, request.question)
  const currentInputs = $clarifyInputs.get()
  const pendingInput = currentInputs[questionKey]

  if (idKey !== questionKey && pendingInput) {
    const { [questionKey]: _removed, ...rest } = currentInputs

    $clarifyInputs.set({ ...rest, [idKey]: currentInputs[idKey] ?? pendingInput })
  }

  $clarifyRequest.set(request)
}

export function clearClarifyRequest(requestId?: string): void {
  const current = $clarifyRequest.get()

  if (!current) {
    return
  }

  if (requestId && current.requestId !== requestId) {
    return
  }

  clearClarifyInput(clarifyInputKey(current.requestId, current.question))
  clearClarifyInput(clarifyInputKey(null, current.question))
  $clarifyRequest.set(null)
}

export function clearClarifyInput(key: string): void {
  const current = $clarifyInputs.get()

  if (!current[key]) {
    return
  }

  const { [key]: _cleared, ...rest } = current

  $clarifyInputs.set(rest)
}

export function setClarifyDraft(key: string, draft: string, position?: ClarifyTextareaPosition): void {
  updateClarifyInput(key, { draft, ...position })
}

export function setClarifyTyping(key: string, typing: boolean): void {
  updateClarifyInput(key, { typing })
}

export function setClarifyFocusLocked(key: string, focusLocked: boolean): void {
  updateClarifyInput(key, { focusLocked })
}

export function setClarifyTextareaPosition(key: string, position: ClarifyTextareaPosition): void {
  updateClarifyInput(key, position)
}

import { afterEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, setSessionArchived } from './hermes'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function installApiMock() {
  const api = vi.fn(async () => ({ ok: true }))

  vi.stubGlobal('window', {})
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api }
  })

  return api
}

describe('Hermes session API wrappers', () => {
  it('routes archive mutations to the owning profile', async () => {
    const api = installApiMock()

    await setSessionArchived('sess-1', true, 'worker-alpha')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/sess-1',
      method: 'PATCH',
      body: { archived: true, profile: 'worker-alpha' }
    })
  })

  it('routes delete mutations to the owning profile', async () => {
    const api = installApiMock()

    await deleteSession('sess/1', 'worker alpha')

    expect(api).toHaveBeenCalledWith({
      path: '/api/sessions/sess%2F1?profile=worker%20alpha',
      method: 'DELETE'
    })
  })
})

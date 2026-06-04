import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS,
  AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS,
  audioSpeakRequestTimeoutMs,
  speakText
} from './hermes'

const originalHermesDesktop = window.hermesDesktop

describe('Hermes desktop API helpers', () => {
  afterEach(() => {
    vi.restoreAllMocks()

    if (originalHermesDesktop) {
      window.hermesDesktop = originalHermesDesktop
    } else {
      delete (window as Partial<Window>).hermesDesktop
    }
  })

  it('bounds blocking TTS synthesis timeouts by text length', () => {
    expect(audioSpeakRequestTimeoutMs('short message')).toBe(AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(8_000))).toBe(280_000)
    expect(audioSpeakRequestTimeoutMs('x'.repeat(100_000))).toBe(AUDIO_SPEAK_MAX_REQUEST_TIMEOUT_MS)
  })

  it('uses an extended timeout for blocking TTS synthesis', async () => {
    const api = vi.fn().mockResolvedValue({
      data_url: 'data:audio/mpeg;base64,AA==',
      mime_type: 'audio/mpeg',
      ok: true,
      provider: 'openai'
    })

    window.hermesDesktop = {
      api
    } as unknown as Window['hermesDesktop']

    await expect(speakText('Read this aloud')).resolves.toEqual({
      data_url: 'data:audio/mpeg;base64,AA==',
      mime_type: 'audio/mpeg',
      ok: true,
      provider: 'openai'
    })

    expect(api).toHaveBeenCalledWith({
      body: { text: 'Read this aloud' },
      method: 'POST',
      path: '/api/audio/speak',
      timeoutMs: AUDIO_SPEAK_MIN_REQUEST_TIMEOUT_MS
    })
  })
})

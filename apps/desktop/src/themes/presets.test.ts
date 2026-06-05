import { describe, expect, it } from 'vitest'

import { BUILTIN_THEMES, hepburnTheme, nousTheme } from './presets'

describe('desktop theme presets', () => {
  it('ports Hepburn colors from the Hermes WebUI skin', () => {
    expect(BUILTIN_THEMES.hepburn).toBe(hepburnTheme)
    expect(hepburnTheme.description).toContain('Hermes WebUI')

    expect(hepburnTheme.colors).toMatchObject({
      background: '#fff3f7',
      foreground: '#3d1a28',
      primary: '#d44a7a',
      sidebarBackground: '#fbe4ed'
    })

    expect(hepburnTheme.darkColors).toMatchObject({
      background: '#110a0f',
      foreground: '#f2e4ee',
      card: '#241420',
      primary: '#f278ad',
      sidebarBackground: '#1e0f19'
    })

    expect(hepburnTheme.typography).toEqual(nousTheme.typography)
  })
})

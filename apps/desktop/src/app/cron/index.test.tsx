import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getCronJobs = vi.fn()

vi.mock('@/hermes', () => ({
  createCronJob: vi.fn(),
  deleteCronJob: vi.fn(),
  getCronJobs: () => getCronJobs(),
  pauseCronJob: vi.fn(),
  resumeCronJob: vi.fn(),
  triggerCronJob: vi.fn(),
  updateCronJob: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('./history-dialog', () => ({
  CronHistoryDialog: ({ job }: { job: null | { id: string } }) =>
    job ? <div role="dialog">History for {job.id}</div> : null
}))

beforeEach(() => {
  getCronJobs.mockResolvedValue([
    {
      enabled: true,
      id: 'daily-briefing',
      name: 'Daily briefing',
      prompt: 'Summarize the day',
      schedule: { expr: '0 9 * * *' },
      state: 'scheduled'
    }
  ])
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('CronView message preview', () => {
  it('places the preview action before the actions menu and opens the history dialog', async () => {
    const { CronView } = await import('./index')

    render(
      <MemoryRouter>
        <CronView onClose={() => {}} />
      </MemoryRouter>
    )

    const preview = await screen.findByRole('button', { name: 'Preview messages for Daily briefing' })
    const actions = screen.getByRole('button', { name: 'Actions for Daily briefing' })

    expect(preview.compareDocumentPosition(actions) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    fireEvent.click(preview)

    await waitFor(() => expect(screen.getByRole('dialog').textContent).toContain('daily-briefing'))
  })
})

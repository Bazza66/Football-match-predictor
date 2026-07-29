import { test, expect } from '@playwright/test'

test.describe('User scenarios (E2E)', () => {
  test('SU-01: single match prediction', async ({ page }) => {
    await page.goto('/')

    await page.getByTestId('tab-train').click()
    const leagueSelect = page.getByTestId('league-select')
    await leagueSelect.selectOption({ index: 1 })

    const seasonSelect = page.getByTestId('season-select')
    await expect(seasonSelect).toBeEnabled()
    await seasonSelect.selectOption({ index: 1 })

    await expect(page.getByTestId('matches-count')).not.toHaveText(
      'Showing 0 matches',
    )

    await page.getByTestId('tab-predict').click()
    await expect(page).toHaveURL(/#predict/)

    await page.getByTestId('predict-button').click()
    await expect(page.getByTestId('prediction-result')).toBeVisible()
    await expect(page.getByTestId('prediction-chart')).toBeVisible()

    await page.getByTestId('history-load').click()
    await expect(
      page.locator('[data-testid="history-table"] tbody tr').first(),
    ).toBeVisible()
  })

  test('SU-02: history analysis with filters and details', async ({ page }) => {
    await page.goto('/')

    const leagueSelect = page.getByTestId('league-select')
    await leagueSelect.selectOption({ index: 1 })

    await page.getByTestId('tab-predict').click()
    await page.getByTestId('history-load').click()

    const firstRow = page
      .locator('[data-testid="history-table"] tbody tr')
      .first()
    await expect(firstRow).toBeVisible()

    const matchCell = firstRow.locator('td').first()
    const matchText = (await matchCell.textContent()) || ''
    const teamToken = matchText.split(' ')[0]
    if (teamToken) {
      await page.getByTestId('history-filter').fill(teamToken)
      await expect(page.getByText(matchText)).toBeVisible()
    }

    await firstRow.getByTestId('history-detail-button').click()
    await expect(page.getByTestId('history-details')).toBeVisible()
  })

  test('SU-03: data error shows message and allows retry', async ({ page }) => {
    let first = true
    await page.route('**/predict**', async (route) => {
      if (first) {
        first = false
        await route.fulfill({
          status: 400,
          body: JSON.stringify({ detail: 'Missing feature row' }),
          contentType: 'application/json',
        })
        return
      }
      await route.continue()
    })

    await page.goto('/')
    await page.getByTestId('tab-predict').click()
    const leagueSelect = page.getByTestId('league-select')
    await leagueSelect.selectOption({ index: 1 })
    await expect(page.getByTestId('matches-count')).not.toHaveText(
      'Showing 0 matches',
    )

    await page.getByTestId('predict-button').click()
    await expect(page.getByText('Failed to generate prediction.')).toBeVisible()

    await page.getByTestId('predict-button').click()
    await expect(page.getByTestId('prediction-result')).toBeVisible()
  })
})

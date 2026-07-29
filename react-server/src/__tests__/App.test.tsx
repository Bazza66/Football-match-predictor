/// <reference types="@testing-library/jest-dom" />
/// <reference types="vitest/globals" />

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, beforeEach, expect, vi } from 'vitest'

import App from '../App'

const leagues = [{ id: 1, code: 'en.1', name: 'Premier League' }]
const seasons = [
  { id: 10, league_id: 1, label: '2010-11' },
  { id: 11, league_id: 1, label: '2011-12' },
]
const matchesAll = [
  {
    id: 1,
    season_id: 10,
    round: '1',
    date: '2010-08-14',
    time: null,
    home_team: 'Arsenal',
    away_team: 'Chelsea',
    home_score: 2,
    away_score: 1,
  },
  {
    id: 2,
    season_id: 11,
    round: '2',
    date: '2011-08-20',
    time: null,
    home_team: 'Liverpool',
    away_team: 'Everton',
    home_score: 1,
    away_score: 1,
  },
]

const predictionPayload = {
  match_id: 1,
  model_name: 'main_xgboost',
  model_version: '20260207_011921_nogit',
  predicted_home_score: null,
  predicted_away_score: null,
  win_prob_home: 0.6,
  win_prob_draw: 0.25,
  win_prob_away: 0.15,
}

const historyPayload = [
  predictionPayload,
  {
    match_id: 2,
    model_name: 'main_xgboost',
    model_version: '20260207_011921_nogit',
    predicted_home_score: null,
    predicted_away_score: null,
    win_prob_home: 0.4,
    win_prob_draw: 0.3,
    win_prob_away: 0.3,
  },
]

const buildJsonResponse = (data: unknown, ok = true) =>
  Promise.resolve({
    ok,
    json: async () => data,
    text: async () => JSON.stringify(data),
  })

describe('App', () => {
  beforeEach(() => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.includes('/health')) {
        return buildJsonResponse({ status: 'ok' })
      }
      if (url.includes('/dataset-integrity')) {
        return buildJsonResponse([])
      }
      if (url.includes('/label-audits')) {
        return buildJsonResponse([])
      }
      if (url.includes('/training-divergence')) {
        return buildJsonResponse([])
      }
      if (url.includes('/model-runs/top')) {
        return buildJsonResponse([])
      }
      if (url.includes('/model-runs/best')) {
        return buildJsonResponse([])
      }
      if (url.includes('/model-runs')) {
        return buildJsonResponse([])
      }
      if (url.includes('/leagues')) {
        return buildJsonResponse(leagues)
      }
      if (url.includes('/seasons')) {
        return buildJsonResponse(seasons)
      }
      if (url.includes('/matches')) {
        if (url.includes('season=2010-11')) {
          return buildJsonResponse([matchesAll[0]])
        }
        return buildJsonResponse(matchesAll)
      }
      if (url.includes('/predict')) {
        return buildJsonResponse(predictionPayload)
      }
      if (url.includes('/predictions')) {
        return buildJsonResponse(historyPayload)
      }
      return buildJsonResponse([])
    })

    vi.stubGlobal('fetch', fetchMock)
  })

  it('updates matches when season changes', async () => {
    render(<App />)
    const leagueSelect = await screen.findByTestId('league-select')
    await userEvent.selectOptions(leagueSelect, 'en.1')

    const seasonSelect = await screen.findByTestId('season-select')
    await userEvent.selectOptions(seasonSelect, '2010-11')

    await waitFor(() => {
      expect(screen.getByTestId('matches-count')).toHaveTextContent('Showing 1 matches')
    })
  })

  it('shows prediction result with probabilities', async () => {
    render(<App />)
    const leagueSelect = await screen.findByTestId('league-select')
    await userEvent.selectOptions(leagueSelect, 'en.1')

    const predictButton = await screen.findByTestId('predict-button')
    await userEvent.click(predictButton)

    const result = await screen.findByTestId('prediction-result')
    expect(result).toHaveTextContent('Prediction')
    expect(screen.getByTestId('prediction-probs')).toHaveTextContent('1 (Home)')
    expect(screen.getByTestId('prediction-probs')).toHaveTextContent('X (Draw)')
    expect(screen.getByTestId('prediction-probs')).toHaveTextContent('2 (Away)')
  })

  it('filters history by team name', async () => {
    render(<App />)
    const leagueSelect = await screen.findByTestId('league-select')
    await userEvent.selectOptions(leagueSelect, 'en.1')

    const loadButton = await screen.findByTestId('history-load')
    await userEvent.click(loadButton)

    await screen.findByText('Arsenal vs Chelsea')
    await screen.findByText('Liverpool vs Everton')

    const filterInput = await screen.findByTestId('history-filter')
    await userEvent.type(filterInput, 'Arsenal')

    await waitFor(() => {
      expect(screen.getByText('Arsenal vs Chelsea')).toBeInTheDocument()
      expect(screen.queryByText('Liverpool vs Everton')).toBeNull()
    })
  })
})

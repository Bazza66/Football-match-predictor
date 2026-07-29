import { useEffect, useMemo, useState } from 'react'
import './App.css'

type League = {
  id: number
  code: string
  name: string
}

type Season = {
  id: number
  league_id: number
  label: string
}

type Match = {
  id: number
  season_id: number
  round: string | null
  date: string | null
  time: string | null
  home_team: string
  away_team: string
  home_score: number | null
  away_score: number | null
}

type PredictionRecord = {
  match_id: number | null
  model_name: string
  model_version: string
  predicted_home_score: number | null
  predicted_away_score: number | null
  win_prob_home: number | null
  win_prob_draw: number | null
  win_prob_away: number | null
}

type ModelArtifact = {
  artifact_type: string
  path: string
  checksum: string
  size_bytes: number
}

type ModelRun = {
  run_id: string
  model_name: string
  model_version: string
  feature_version: string
  commit_hash: string | null
  league_code: string | null
  season_label: string | null
  status: string
  metrics: Record<string, unknown> | null
  params: Record<string, unknown> | null
  data_summary: Record<string, unknown> | null
  created_at: string
  artifacts: ModelArtifact[] | null
}

type ModelRunSummary = {
  run_id: string
  model_name: string
  model_version: string
  feature_version: string
  league_code: string | null
  season_label: string | null
  metric: number | null
  metric_name: string | null
  created_at: string
}

type BestRunEntry = {
  run_id: string | null
  model_name: string
  model_version: string
  feature_version: string
  league_code: string | null
  season_label: string | null
  metric_name: string | null
  metric_value: number | null
  direction: string | null
  updated_at: string | null
}

type DatasetIntegrityReport = {
  report_id: string
  created_at: string
  summary_url: string
  missingness_url: string | null
  golden_sample_url: string | null
}

type LabelAuditReport = {
  report_id: string
  run_id: string
  created_at: string
  report_url: string
  chart_url: string | null
  confusion_url: string | null
  league_code: string | null
  season_label: string | null
}

type TrainingDivergenceReport = {
  report_id: string
  run_id: string
  created_at: string
  report_url: string
  loss_chart_url: string | null
  grad_chart_url: string | null
  league_code: string | null
  season_label: string | null
}

type ModelEvalReport = {
  report_id: string
  run_id: string | null
  created_at: string
  status: string
  report_url: string | null
  metrics_json_url: string | null
  metrics_csv_url: string | null
  confusion_url: string | null
  confusion_baseline_url: string | null
  calibration_url: string | null
  entropy_hist_url: string | null
  max_prob_hist_url: string | null
  subsets_url: string | null
  league_code: string | null
  season_label: string | null
  model_name: string | null
  model_version: string | null
  baseline_name: string | null
  baseline_version: string | null
  feature_version: string | null
  error: string | null
}

type TuningSweepReport = {
  report_id: string
  created_at: string
  status: string
  started_at: string | null
  updated_at: string | null
  percent: number | null
  total_iterations: number | null
  completed_iterations: number | null
  current_league: string | null
  current_iteration: number | null
  elapsed_seconds: number | null
  eta_seconds: number | null
  errors_count: number | null
  trainer: string | null
  model: string | null
  feature_version: string | null
  season_label: string | null
  league_codes: string[] | null
  n_iter: number | null
  seed: number | null
  train_ratio: number | null
  val_ratio: number | null
  class_weight: string | null
  calibration: boolean | null
  calibration_method: string | null
  n_estimators: number | null
  iterations: number | null
  fast_mode: boolean | null
  results_csv_url: string | null
  results_tsv_url: string | null
  results_md_url: string | null
  summary_by_league_url: string | null
  best_by_league_url: string | null
  metadata_url: string | null
  errors_url: string | null
  error: string | null
}

type RunListMode = 'top' | 'recent'
type FlowTab = 'train' | 'predict' | 'reports' | 'audit'
type FlowView = FlowTab | 'all'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const PREDICTION_MODEL = 'main_xgboost'

type RetryOptions = {
  retries: number
  minDelayMs: number
  maxDelayMs: number
  timeoutMs: number
  retryOnStatuses: number[]
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  retries: 2,
  minDelayMs: 300,
  maxDelayMs: 2000,
  timeoutMs: 5000,
  retryOnStatuses: [408, 429, 502, 503, 504],
}

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms)
  })

const fetchWithRetry = async (
  input: RequestInfo | URL,
  init?: RequestInit,
  overrides: Partial<RetryOptions> = {},
) => {
  const options = { ...DEFAULT_RETRY_OPTIONS, ...overrides }
  let lastError: unknown = null

  for (let attempt = 0; attempt <= options.retries; attempt += 1) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs)

    try {
      const response = await fetch(input, { ...init, signal: controller.signal })
      clearTimeout(timeoutId)
      if (!options.retryOnStatuses.includes(response.status)) {
        return response
      }
      if (attempt === options.retries) {
        return response
      }
      lastError = new Error(`Retryable status ${response.status}`)
    } catch (error) {
      clearTimeout(timeoutId)
      lastError = error
      if (attempt === options.retries) {
        throw error
      }
    }

    const baseDelay = Math.min(
      options.maxDelayMs,
      options.minDelayMs * 2 ** attempt,
    )
    const jitter = Math.random() * options.minDelayMs
    await sleep(baseDelay + jitter)
  }

  if (lastError instanceof Error) {
    throw lastError
  }
  throw new Error('Request failed after retries.')
}

const formatDate = (value?: string | null) => {
  if (!value) return 'TBD'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat('en-US', { dateStyle: 'medium' }).format(parsed)
}

const formatDateTime = (value?: string | null) => {
  if (!value) return 'TBD'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed)
}

const formatMatchDateTime = (match?: Match | null) => {
  if (!match?.date) return 'TBD'
  if (!match.time) return formatDate(match.date)
  const iso = `${match.date}T${match.time}`
  return formatDateTime(iso)
}

const formatNumber = (value?: number | null, digits = 4) => {
  if (value == null || Number.isNaN(value)) return '-'
  return value.toFixed(digits)
}

const formatScore = (match: Match) => {
  if (match.home_score == null || match.away_score == null) return '-'
  return `${match.home_score} - ${match.away_score}`
}

const formatProbability = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) return '-'
  return `${(value * 100).toFixed(1)}%`
}

const formatDuration = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) return '-'
  const seconds = Math.max(0, Math.floor(value))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remaining = seconds % 60
  if (minutes < 60) return `${minutes}m ${remaining}s`
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return `${hours}h ${mins}m`
}

const sanitizeTsvValue = (value: unknown) => {
  if (value == null) return ''
  return String(value).replace(/\r?\n/g, ' ').replace(/\t/g, ' ').trim()
}

const buildTsv = (headers: string[], rows: Array<Array<unknown>>) => {
  const lines = [headers, ...rows].map((row) =>
    row.map((value) => sanitizeTsvValue(value)).join('\t'),
  )
  return lines.join('\n')
}

const labelFromProbs = (
  home: number | null,
  draw: number | null,
  away: number | null,
) => {
  const scores = [
    { label: 'H', value: home ?? -1 },
    { label: 'D', value: draw ?? -1 },
    { label: 'A', value: away ?? -1 },
  ]
  scores.sort((a, b) => b.value - a.value)
  return scores[0].label
}

const outcomeLabel = (label: string) => {
  if (label === 'H') return '1'
  if (label === 'D') return 'X'
  if (label === 'A') return '2'
  return label
}

const outcomeFromScore = (match?: Match | null) => {
  if (!match) return null
  if (match.home_score == null || match.away_score == null) return null
  if (match.home_score > match.away_score) return 'H'
  if (match.home_score < match.away_score) return 'A'
  return 'D'
}

const REPORT_METRICS = [
  { key: 'test_log_loss', label: 'Test log loss (lower is better)' },
  { key: 'test_brier_score', label: 'Test Brier score (lower is better)' },
  { key: 'test_f1_macro', label: 'Test F1 macro (higher is better)' },
  { key: 'test_accuracy', label: 'Test accuracy (higher is better)' },
]

const RUN_LIMIT_OPTIONS = [10, 50, 100]
const FLOW_TABS: Array<{ key: FlowView; label: string; description: string }> = [
  { key: 'all', label: 'All', description: 'Full view' },
  { key: 'train', label: 'Train', description: 'All leagues + setup' },
  { key: 'predict', label: 'Predict', description: 'Score matches' },
  { key: 'reports', label: 'Reports', description: 'Runs + evaluation' },
  { key: 'audit', label: 'Audit', description: 'Label, divergence, integrity' },
]
const FLOW_SEQUENCE: FlowTab[] = ['train', 'predict', 'reports', 'audit']
const FLOW_TAB_KEYS: FlowView[] = FLOW_TABS.map((tab) => tab.key)

const MODEL_DESCRIPTIONS: Record<string, string> = {
  baseline_classifier:
    'Baseline multinomial logistic regression with scaling and median imputation.',
  main_xgboost:
    'Main model using gradient-boosted trees (XGBoost) for multiclass outcomes.',
  main_catboost:
    'Main model using CatBoost gradient boosting tuned for categorical robustness.',
}

type ApiStatus = 'checking' | 'ok' | 'fail'

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [activeTab, setActiveTab] = useState<FlowView>('all')
  const [copiedKey, setCopiedKey] = useState<string>('')
  const [copyMessage, setCopyMessage] = useState<string>('')
  const [trainStatus, setTrainStatus] = useState<string>('')
  const [trainRunningKey, setTrainRunningKey] = useState<string>('')
  const [forceRetrain, setForceRetrain] = useState<boolean>(true)
  const [fastTrainIters, setFastTrainIters] = useState<number>(5)
  const [autoFlowNext, setAutoFlowNext] = useState<FlowTab | null>(null)
  const [autoFlowQueuedAt, setAutoFlowQueuedAt] = useState<number | null>(null)
  const [autoFlowWatchingRuns, setAutoFlowWatchingRuns] = useState<boolean>(false)
  const [flowRunFilterSince, setFlowRunFilterSince] = useState<number | null>(null)
  const [flowRunFilterActive, setFlowRunFilterActive] = useState<boolean>(false)
  const [leagues, setLeagues] = useState<League[]>([])
  const [seasons, setSeasons] = useState<Season[]>([])
  const [matches, setMatches] = useState<Match[]>([])
  const [selectedMatchId, setSelectedMatchId] = useState<string>('')
  const [predictionResult, setPredictionResult] =
    useState<PredictionRecord | null>(null)
  const [predictionStatus, setPredictionStatus] = useState<string>('')
  const [predictionLoading, setPredictionLoading] = useState<boolean>(false)
  const [historyRows, setHistoryRows] = useState<PredictionRecord[]>([])
  const [historyFilter, setHistoryFilter] = useState<string>('')
  const [historyStatus, setHistoryStatus] = useState<string>('')
  const [historyLoading, setHistoryLoading] = useState<boolean>(false)
  const [selectedHistoryRecord, setSelectedHistoryRecord] =
    useState<PredictionRecord | null>(null)
  const [modelRuns, setModelRuns] = useState<ModelRun[]>([])
  const [topRuns, setTopRuns] = useState<ModelRunSummary[]>([])
  const [bestRuns, setBestRuns] = useState<BestRunEntry[]>([])
  const [selectedLeague, setSelectedLeague] = useState<string>('')
  const [selectedSeason, setSelectedSeason] = useState<string>('')
  const [status, setStatus] = useState<string>('')
  const [modelStatus, setModelStatus] = useState<string>('')
  const [integrityStatus, setIntegrityStatus] = useState<string>('')
  const [selectedRunId, setSelectedRunId] = useState<string>('')
  const [reportMetric, setReportMetric] = useState<string>('test_log_loss')
  const [runListMode, setRunListMode] = useState<RunListMode>('top')
  const [runListLimit, setRunListLimit] = useState<number>(10)
  const [integrityReports, setIntegrityReports] = useState<DatasetIntegrityReport[]>(
    [],
  )
  const [selectedIntegrityId, setSelectedIntegrityId] = useState<string>('')
  const [integrityMarkdown, setIntegrityMarkdown] = useState<string>('')
  const [labelAuditStatus, setLabelAuditStatus] = useState<string>('')
  const [labelAudits, setLabelAudits] = useState<LabelAuditReport[]>([])
  const [selectedLabelAuditId, setSelectedLabelAuditId] = useState<string>('')
  const [labelAuditMarkdown, setLabelAuditMarkdown] = useState<string>('')
  const [filterAuditsToRun, setFilterAuditsToRun] = useState<boolean>(true)
  const [labelAuditRunning, setLabelAuditRunning] = useState<boolean>(false)
  const [divergenceStatus, setDivergenceStatus] = useState<string>('')
  const [divergenceReports, setDivergenceReports] = useState<
    TrainingDivergenceReport[]
  >([])
  const [selectedDivergenceId, setSelectedDivergenceId] =
    useState<string>('')
  const [divergenceMarkdown, setDivergenceMarkdown] = useState<string>('')
  const [filterDivergenceToRun, setFilterDivergenceToRun] =
    useState<boolean>(true)
  const [divergenceRunning, setDivergenceRunning] = useState<boolean>(false)
  const [modelEvalReports, setModelEvalReports] = useState<ModelEvalReport[]>([])
  const [selectedModelEvalId, setSelectedModelEvalId] = useState<string>('')
  const [modelEvalMarkdown, setModelEvalMarkdown] = useState<string>('')
  const [modelEvalStatus, setModelEvalStatus] = useState<string>('')
  const [modelEvalRunStatus, setModelEvalRunStatus] = useState<string>('')
  const [modelEvalRunning, setModelEvalRunning] = useState<boolean>(false)
  const [tuningSweepReports, setTuningSweepReports] = useState<
    TuningSweepReport[]
  >([])
  const [selectedTuningSweepId, setSelectedTuningSweepId] =
    useState<string>('')
  const [tuningSweepStatus, setTuningSweepStatus] = useState<string>('')
  const [tuningSweepRunStatus, setTuningSweepRunStatus] = useState<string>('')
  const [tuningSweepRunning, setTuningSweepRunning] =
    useState<boolean>(false)
  const [tuningSweepIters, setTuningSweepIters] = useState<number>(300)
  const [subfolderStatus, setSubfolderStatus] = useState<string>('')
  const [subfolderActionKey, setSubfolderActionKey] = useState<string>('')

  const loadModelEvalReports = (preferredId?: string) => {
    const params = new URLSearchParams()
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)
    const query = params.toString()
    const url = query ? `${API_URL}/model-eval?${query}` : `${API_URL}/model-eval`
    fetchWithRetry(url)
      .then((res) => res.json())
      .then((data: ModelEvalReport[]) => {
        if (!Array.isArray(data)) {
          setModelEvalReports([])
          return
        }
        setModelEvalReports(data)
        setModelEvalStatus('')
        setSelectedModelEvalId((current) => {
          if (data.length === 0) return ''
          const target = preferredId || current
          if (target && data.find((report) => report.report_id === target)) {
            return target
          }
          return data[0].report_id
        })
      })
      .catch(() =>
        setModelEvalStatus('Failed to load model evaluation reports.'),
      )
  }

  const loadTuningSweepReports = (preferredId?: string) => {
    const params = new URLSearchParams()
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)
    const query = params.toString()
    const url = query
      ? `${API_URL}/tuning-sweep?${query}`
      : `${API_URL}/tuning-sweep`
    fetchWithRetry(url)
      .then((res) => res.json())
      .then((data: TuningSweepReport[]) => {
        if (!Array.isArray(data)) {
          setTuningSweepReports([])
          return
        }
        setTuningSweepReports(data)
        setTuningSweepStatus('')
        setSelectedTuningSweepId((current) => {
          if (data.length === 0) return ''
          const target = preferredId || current
          if (target && data.find((report) => report.report_id === target)) {
            return target
          }
          return data[0].report_id
        })
      })
      .catch(() => setTuningSweepStatus('Failed to load tuning sweeps.'))
  }

  const loadLabelAudits = (preferredId?: string) => {
    const params = new URLSearchParams()
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)
    fetchWithRetry(`${API_URL}/label-audits?${params.toString()}`)
      .then((res) => res.json())
      .then((data: LabelAuditReport[]) => {
        if (!Array.isArray(data)) {
          setLabelAudits([])
          return
        }
        setLabelAudits(data)
        setLabelAuditStatus('')
        setSelectedLabelAuditId((current) => {
          if (data.length === 0) return ''
          const target = preferredId || current
          if (target && data.find((report) => report.report_id === target)) {
            return target
          }
          return data[0].report_id
        })
      })
      .catch(() => setLabelAuditStatus('Failed to load label audits.'))
  }

  const loadDivergenceReports = (preferredId?: string) => {
    const params = new URLSearchParams()
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)
    fetchWithRetry(`${API_URL}/training-divergence?${params.toString()}`)
      .then((res) => res.json())
      .then((data: TrainingDivergenceReport[]) => {
        if (!Array.isArray(data)) {
          setDivergenceReports([])
          return
        }
        setDivergenceReports(data)
        setDivergenceStatus('')
        setSelectedDivergenceId((current) => {
          if (data.length === 0) return ''
          const target = preferredId || current
          if (target && data.find((report) => report.report_id === target)) {
            return target
          }
          return data[0].report_id
        })
      })
      .catch(() => setDivergenceStatus('Failed to load divergence reports.'))
  }

  useEffect(() => {
    fetchWithRetry(`${API_URL}/health`)
      .then((res) => {
        setApiStatus(res.ok ? 'ok' : 'fail')
      })
      .catch(() => setApiStatus('fail'))
  }, [])

  useEffect(() => {
    const hash = window.location.hash.replace('#', '')
    if (FLOW_TAB_KEYS.includes(hash as FlowView)) {
      setActiveTab(hash as FlowView)
    }
  }, [])

  useEffect(() => {
    const nextHash = `#${activeTab}`
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, '', nextHash)
    }
  }, [activeTab])

  useEffect(() => {
    fetchWithRetry(`${API_URL}/dataset-integrity`)
      .then((res) => res.json())
      .then((data: DatasetIntegrityReport[]) => {
        if (!Array.isArray(data)) {
          setIntegrityReports([])
          return
        }
        setIntegrityReports(data)
        setIntegrityStatus('')
        setSelectedIntegrityId((current) => {
          if (data.length === 0) return ''
          if (!current || !data.find((report) => report.report_id === current)) {
            return data[0].report_id
          }
          return current
        })
      })
      .catch(() => setIntegrityStatus('Failed to load integrity reports.'))
  }, [])

  useEffect(() => {
    loadLabelAudits()
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    loadDivergenceReports()
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    loadModelEvalReports()
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    loadTuningSweepReports()
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    fetchWithRetry(`${API_URL}/leagues`)
      .then((res) => res.json())
      .then((data) => {
        setLeagues(data)
        setStatus('')
      })
      .catch(() => setStatus('Failed to load leagues.'))
  }, [])

  useEffect(() => {
    if (!selectedLeague) {
      setSeasons([])
      setSelectedSeason('')
      return
    }

    fetchWithRetry(`${API_URL}/seasons?league=${encodeURIComponent(selectedLeague)}`)
      .then((res) => res.json())
      .then((data) => {
        setSeasons(data)
        setStatus('')
      })
      .catch(() => setStatus('Failed to load seasons.'))
  }, [selectedLeague])

  useEffect(() => {
    if (!selectedSeason) return
    if (seasons.length === 0) return
    const exists = seasons.some((season) => season.label === selectedSeason)
    if (!exists) {
      setSelectedSeason('')
      setStatus('Selected season not available for this league. Showing all seasons.')
    }
  }, [seasons, selectedSeason])

  useEffect(() => {
    if (!selectedLeague) {
      setMatches([])
      return
    }

    const params = new URLSearchParams({ league: selectedLeague, limit: '200' })
    if (selectedSeason) params.set('season', selectedSeason)

    fetchWithRetry(`${API_URL}/matches?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => {
        setMatches(data)
        setStatus('')
      })
      .catch(() => setStatus('Failed to load matches.'))
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    if (matches.length === 0) {
      setSelectedMatchId('')
      return
    }
    const exists = matches.some((match) => String(match.id) === selectedMatchId)
    if (!exists) {
      setSelectedMatchId(String(matches[0].id))
    }
  }, [matches, selectedMatchId])

  const loadModelRuns = async (
    options: { preferredId?: string; forceSelect?: boolean } = {},
  ) => {
    const params = new URLSearchParams({ limit: '500' })
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)

    try {
      const res = await fetchWithRetry(
        `${API_URL}/model-runs?${params.toString()}`,
      )
      const payload = (await res.json()) as ModelRun[]
      const runs = Array.isArray(payload) ? payload : []
      setModelRuns(runs)
      setModelStatus('')
      setSelectedRunId((current) => {
        if (runs.length === 0) return ''
        if (
          options.preferredId &&
          runs.find((run) => run.run_id === options.preferredId)
        ) {
          return options.preferredId
        }
        if (options.forceSelect) {
          return runs[0].run_id
        }
        if (!current || !runs.find((run) => run.run_id === current)) {
          return runs[0].run_id
        }
        return current
      })
      return runs
    } catch (error) {
      setModelStatus('Failed to load model runs.')
      return []
    }
  }

  useEffect(() => {
    loadModelRuns()
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    if (!autoFlowWatchingRuns || autoFlowQueuedAt == null) return
    let cancelled = false

    const poll = async () => {
      const runs = await loadModelRuns()
      if (cancelled) return
      const nextRun = runs.find((run) => {
        const createdAt = Date.parse(run.created_at)
        return Number.isFinite(createdAt) && createdAt >= autoFlowQueuedAt
      })
      if (nextRun) {
        selectRun(nextRun.run_id, nextRun)
        setActiveTab('predict')
        setTrainStatus('New runs detected. Moving to Predict.')
        setAutoFlowWatchingRuns(false)
        return
      }
      const elapsed = Date.now() - autoFlowQueuedAt
      if (elapsed > 30000) {
        setFlowRunFilterSince(null)
        setFlowRunFilterActive(false)
        setActiveTab('predict')
        setTrainStatus(
          'No new runs detected (likely skipped). Showing existing runs.',
        )
        setAutoFlowWatchingRuns(false)
      }
    }

    poll()
    const intervalId = window.setInterval(poll, 7000)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [autoFlowWatchingRuns, autoFlowQueuedAt, selectedLeague, selectedSeason])

  useEffect(() => {
    const params = new URLSearchParams()
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)

    fetchWithRetry(`${API_URL}/model-runs/best?${params.toString()}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Best runs request failed (${res.status})`)
        }
        return res.json()
      })
      .then((data: BestRunEntry[]) => {
        setBestRuns(Array.isArray(data) ? data : [])
      })
      .catch(() => setBestRuns([]))
  }, [selectedLeague, selectedSeason])

  useEffect(() => {
    if (runListMode !== 'top') {
      setTopRuns([])
      return
    }

    const params = new URLSearchParams({
      limit: String(runListLimit),
      metric: reportMetric,
    })
    if (selectedLeague) params.set('league', selectedLeague)
    if (selectedSeason) params.set('season', selectedSeason)

    fetchWithRetry(`${API_URL}/model-runs/top?${params.toString()}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Top runs request failed (${res.status})`)
        }
        return res.json()
      })
      .then((data: ModelRunSummary[]) => {
        setTopRuns(Array.isArray(data) ? data : [])
        setModelStatus('')
      })
      .catch(() => setModelStatus('Failed to load top model runs.'))
  }, [selectedLeague, selectedSeason, reportMetric, runListMode, runListLimit])

  useEffect(() => {
    const report = integrityReports.find(
      (item) => item.report_id === selectedIntegrityId,
    )
    if (!report?.summary_url) {
      setIntegrityMarkdown('')
      return
    }
    const url = report.summary_url.startsWith('http')
      ? report.summary_url
      : `${API_URL}${report.summary_url}`
    fetchWithRetry(url)
      .then((res) => res.text())
      .then((text) => {
        setIntegrityMarkdown(text)
        setIntegrityStatus('')
      })
      .catch(() => setIntegrityStatus('Failed to load integrity summary.'))
  }, [integrityReports, selectedIntegrityId])

  const flowFilterEnabled =
    flowRunFilterSince !== null &&
    flowRunFilterActive &&
    (activeTab === 'predict' || activeTab === 'reports' || activeTab === 'audit')

  const flowFilteredRuns = useMemo(() => {
    if (flowRunFilterSince == null) return modelRuns
    return modelRuns.filter((run) => {
      const createdAt = Date.parse(run.created_at)
      return Number.isFinite(createdAt) && createdAt >= flowRunFilterSince
    })
  }, [modelRuns, flowRunFilterSince])

  const flowRunIds = useMemo(() => {
    return new Set(flowFilteredRuns.map((run) => run.run_id))
  }, [flowFilteredRuns])

  const visibleModelRuns = flowFilterEnabled ? flowFilteredRuns : modelRuns

  const applyRunScope = (
    run?: { league_code: string | null; season_label: string | null } | null,
  ) => {
    if (!run) return
    if (run.league_code) {
      setSelectedLeague(run.league_code)
    }
    if (run.season_label) {
      setSelectedSeason(run.season_label)
      return
    }
    if (run.league_code) {
      setSelectedSeason('')
    }
  }

  const selectRun = (
    runId: string,
    run?: { league_code: string | null; season_label: string | null } | null,
  ) => {
    setSelectedRunId(runId)
    const resolved =
      run ||
      visibleModelRuns.find((item) => item.run_id === runId) ||
      modelRuns.find((item) => item.run_id === runId)
    applyRunScope(resolved ?? null)
  }

  const clearScope = () => {
    setSelectedLeague('')
    setSelectedSeason('')
  }

  useEffect(() => {
    if (!flowFilterEnabled) return
    if (visibleModelRuns.length === 0) {
      setSelectedRunId('')
      return
    }
    if (!visibleModelRuns.find((run) => run.run_id === selectedRunId)) {
      selectRun(visibleModelRuns[0].run_id, visibleModelRuns[0])
    }
  }, [flowFilterEnabled, visibleModelRuns, selectedRunId])

  const filteredLabelAudits = useMemo(() => {
    let audits = labelAudits
    if (flowFilterEnabled) {
      audits = audits.filter((report) => flowRunIds.has(report.run_id))
    }
    if (!filterAuditsToRun || !selectedRunId) return audits
    return audits.filter((report) => report.run_id === selectedRunId)
  }, [filterAuditsToRun, labelAudits, selectedRunId, flowFilterEnabled, flowRunIds])

  const filteredDivergenceReports = useMemo(() => {
    let reports = divergenceReports
    if (flowFilterEnabled) {
      reports = reports.filter((report) => flowRunIds.has(report.run_id))
    }
    if (!filterDivergenceToRun || !selectedRunId) return reports
    return reports.filter((report) => report.run_id === selectedRunId)
  }, [
    filterDivergenceToRun,
    divergenceReports,
    selectedRunId,
    flowFilterEnabled,
    flowRunIds,
  ])

  useEffect(() => {
    const report = filteredLabelAudits.find(
      (item) => item.report_id === selectedLabelAuditId,
    )
    if (!report?.report_url) {
      setLabelAuditMarkdown('')
      return
    }
    const url = report.report_url.startsWith('http')
      ? report.report_url
      : `${API_URL}${report.report_url}`
    fetchWithRetry(url)
      .then((res) => res.text())
      .then((text) => {
        setLabelAuditMarkdown(text)
        setLabelAuditStatus('')
      })
      .catch(() => setLabelAuditStatus('Failed to load label audit summary.'))
  }, [filteredLabelAudits, selectedLabelAuditId])

  useEffect(() => {
    const report = filteredDivergenceReports.find(
      (item) => item.report_id === selectedDivergenceId,
    )
    if (!report?.report_url) {
      setDivergenceMarkdown('')
      return
    }
    const url = report.report_url.startsWith('http')
      ? report.report_url
      : `${API_URL}${report.report_url}`
    fetchWithRetry(url)
      .then((res) => res.text())
      .then((text) => {
        setDivergenceMarkdown(text)
        setDivergenceStatus('')
      })
      .catch(() => setDivergenceStatus('Failed to load divergence summary.'))
  }, [filteredDivergenceReports, selectedDivergenceId])

  useEffect(() => {
    if (filteredLabelAudits.length === 0) {
      setSelectedLabelAuditId('')
      return
    }
    if (
      !filteredLabelAudits.find(
        (report) => report.report_id === selectedLabelAuditId,
      )
    ) {
      setSelectedLabelAuditId(filteredLabelAudits[0].report_id)
    }
  }, [filteredLabelAudits, selectedLabelAuditId])

  useEffect(() => {
    if (filteredDivergenceReports.length === 0) {
      setSelectedDivergenceId('')
      return
    }
    if (
      !filteredDivergenceReports.find(
        (report) => report.report_id === selectedDivergenceId,
      )
    ) {
      setSelectedDivergenceId(filteredDivergenceReports[0].report_id)
    }
  }, [filteredDivergenceReports, selectedDivergenceId])

  const filteredModelEvalReports = useMemo(() => {
    if (!flowFilterEnabled) return modelEvalReports
    return modelEvalReports.filter((report) =>
      report.run_id ? flowRunIds.has(report.run_id) : false,
    )
  }, [modelEvalReports, flowFilterEnabled, flowRunIds])

  useEffect(() => {
    const report = filteredModelEvalReports.find(
      (item) => item.report_id === selectedModelEvalId,
    )
    if (!report?.report_url) {
      setModelEvalMarkdown('')
      return
    }
    const url = report.report_url.startsWith('http')
      ? report.report_url
      : `${API_URL}${report.report_url}`
    fetchWithRetry(url)
      .then((res) => res.text())
      .then((text) => {
        setModelEvalMarkdown(text)
        setModelEvalStatus('')
      })
      .catch(() => setModelEvalStatus('Failed to load model eval summary.'))
  }, [filteredModelEvalReports, selectedModelEvalId])

  useEffect(() => {
    if (filteredModelEvalReports.length === 0) {
      setSelectedModelEvalId('')
      return
    }
    if (
      !filteredModelEvalReports.find(
        (report) => report.report_id === selectedModelEvalId,
      )
    ) {
      setSelectedModelEvalId(filteredModelEvalReports[0].report_id)
    }
  }, [filteredModelEvalReports, selectedModelEvalId])

  useEffect(() => {
    if (tuningSweepReports.length === 0) {
      setSelectedTuningSweepId('')
      return
    }
    if (
      !tuningSweepReports.find(
        (report) => report.report_id === selectedTuningSweepId,
      )
    ) {
      setSelectedTuningSweepId(tuningSweepReports[0].report_id)
    }
  }, [tuningSweepReports, selectedTuningSweepId])

  const matchById = useMemo(() => {
    return new Map(matches.map((match) => [match.id, match]))
  }, [matches])

  const selectedMatch = useMemo(() => {
    if (!selectedMatchId) return null
    return matches.find((match) => String(match.id) === selectedMatchId) || null
  }, [matches, selectedMatchId])

  const selectedLeagueName = useMemo(() => {
    const league = leagues.find((item) => item.code === selectedLeague)
    return league?.name || 'All leagues'
  }, [leagues, selectedLeague])

  const leagueNameByCode = useMemo(() => {
    return new Map(leagues.map((league) => [league.code, league.name]))
  }, [leagues])

  const formatLeagueLabel = (code?: string | null) => {
    if (!code) return 'All leagues'
    const name = leagueNameByCode.get(code)
    return name ? `${name} (${code})` : code
  }

  const formatLeagueList = (codes?: string[] | null) => {
    if (!codes || codes.length === 0) return 'All leagues'
    if (codes.length <= 4) return codes.join(', ')
    const head = codes.slice(0, 4).join(', ')
    return `${head} (+${codes.length - 4} more)`
  }

  const resolveReportUrl = (path?: string | null) => {
    if (!path) return null
    return path.startsWith('http') ? path : `${API_URL}${path}`
  }

  const copyText = async (text: string) => {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return
    }
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
  }

  const handleCopy = async (text: string, key: string) => {
    try {
      await copyText(text)
      setCopiedKey(key)
      setCopyMessage('Copied')
    } catch (error) {
      setCopiedKey(key)
      setCopyMessage('Copy failed')
    } finally {
      window.setTimeout(() => {
        setCopiedKey('')
        setCopyMessage('')
      }, 1800)
    }
  }

  const copyLabel = (key: string, fallback = 'Copy') => {
    if (copiedKey !== key) return fallback
    return copyMessage || 'Copied'
  }

  const handleClearFlowFilter = () => {
    setFlowRunFilterSince(null)
    setFlowRunFilterActive(false)
  }

  const renderFlowFilterControls = (
    idSuffix: string,
    asControlGroup = false,
  ) => {
    if (flowRunFilterSince == null) return null
    const toggleId = `flow-filter-${idSuffix}`
    const sinceLabel = formatDateTime(new Date(flowRunFilterSince).toISOString())
    const content = (
      <div className="d-flex align-items-center gap-2 flex-wrap">
        <div className="form-check form-switch mb-0">
          <input
            className="form-check-input"
            type="checkbox"
            id={toggleId}
            checked={flowRunFilterActive}
            onChange={(event) => setFlowRunFilterActive(event.target.checked)}
          />
          <label className="form-check-label" htmlFor={toggleId}>
            Show new runs only
          </label>
        </div>
        <button
          type="button"
          className="btn btn-sm btn-outline-dark"
          onClick={handleClearFlowFilter}
        >
          Clear
        </button>
        <div className="small text-muted">Since {sinceLabel}</div>
      </div>
    )
    if (asControlGroup) {
      return (
        <div className="control-group">
          <label className="form-label">Flow filter</label>
          {content}
        </div>
      )
    }
    return content
  }

  const handleRunTraining = async (options: {
    key: string
    label: string
    trainer: 'main' | 'baseline'
    model: 'xgboost' | 'catboost'
    overrides?: Record<string, string>
    autoFlow?: boolean
  }) => {
    setTrainStatus('')
    setTrainRunningKey(options.key)
    if (!options.autoFlow) {
      setAutoFlowNext(null)
      setAutoFlowQueuedAt(null)
      setAutoFlowWatchingRuns(false)
    }
    if (options.autoFlow) {
      setSelectedLeague('')
      setSelectedSeason('')
    }
    setTrainStatus(`Queuing ${options.label.toLowerCase()} training...`)
    try {
      const params = new URLSearchParams({
        trainer: options.trainer,
        model: options.model,
        version: 'v1',
        calibrate: 'false',
        verbose_eval: '1',
        continue_on_error: 'true',
        skip_existing: forceRetrain ? 'false' : 'true',
      })
      if (options.overrides) {
        for (const [key, value] of Object.entries(options.overrides)) {
          params.set(key, value)
        }
      }
      const res = await fetchWithRetry(
        `${API_URL}/train-per-league/run?${params.toString()}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Training request failed (${res.status})`)
      }
      if (options.autoFlow) {
        const queuedAt = Date.now()
        setAutoFlowQueuedAt(queuedAt)
        setFlowRunFilterSince(queuedAt)
        setFlowRunFilterActive(true)
        setAutoFlowWatchingRuns(true)
        setAutoFlowNext('reports')
        setTrainStatus(
          `Training queued for ${options.label}. Waiting for new runs...`,
        )
      } else {
        setTrainStatus(
          `Training queued for ${options.label}. This may take a while.`,
        )
      }
    } catch (error) {
      setTrainStatus(
        error instanceof Error ? error.message : 'Failed to start training.',
      )
    } finally {
      setTrainRunningKey('')
    }
  }

  const trainButtonLabel = (key: string, fallback: string) => {
    if (trainRunningKey !== key) return fallback
    return 'Queuing...'
  }

  const handleRunModelEval = async () => {
    if (!selectedRun) return
    setModelEvalRunning(true)
    setModelEvalRunStatus('Queuing evaluation run...')
    try {
      const res = await fetchWithRetry(
        `${API_URL}/model-eval/run?run_id=${encodeURIComponent(
          selectedRun.run_id,
        )}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        throw new Error(`Model eval request failed (${res.status})`)
      }
      const data = (await res.json()) as { report_id?: string }
      const reportId = data?.report_id
      setModelEvalRunStatus(
        reportId
          ? `Report queued: ${reportId}`
          : 'Report queued. Refresh to see status.',
      )
      loadModelEvalReports(reportId)
      if (reportId) {
        window.setTimeout(() => loadModelEvalReports(reportId), 5000)
      }
      if (autoFlowNext === 'audit') {
        setActiveTab('audit')
        setAutoFlowNext(null)
      }
    } catch (error) {
      setModelEvalRunStatus('Failed to start model evaluation.')
    } finally {
      setModelEvalRunning(false)
    }
  }

  const resolveSweepTrainer = (modelName?: string | null) => {
    if (modelName === 'baseline_classifier') {
      return { trainer: 'baseline', model: 'xgboost' }
    }
    if (modelName === 'main_catboost') {
      return { trainer: 'main', model: 'catboost' }
    }
    return { trainer: 'main', model: 'xgboost' }
  }

  const handleRunTuningSweep = async () => {
    setTuningSweepRunning(true)
    setTuningSweepRunStatus('Queuing tuning sweep...')
    try {
      const targetModel = selectedRun?.model_name || PREDICTION_MODEL
      const { trainer, model } = resolveSweepTrainer(targetModel)
      const params = new URLSearchParams({
        trainer,
        model,
        version: 'v1',
        n_iter: String(Math.max(1, tuningSweepIters)),
        fast: 'true',
        calibrate: 'false',
        continue_on_error: 'true',
      })
      if (selectedLeague) params.set('league', selectedLeague)
      if (selectedSeason) params.set('season', selectedSeason)
      const res = await fetchWithRetry(
        `${API_URL}/tuning-sweep/run?${params.toString()}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Sweep request failed (${res.status})`)
      }
      const data = (await res.json()) as { report_id?: string }
      const reportId = data?.report_id
      setTuningSweepRunStatus(
        reportId
          ? `Sweep queued: ${reportId}`
          : 'Sweep queued. Refresh to see status.',
      )
      loadTuningSweepReports(reportId)
      if (reportId) {
        window.setTimeout(() => loadTuningSweepReports(reportId), 5000)
      }
    } catch (error) {
      setTuningSweepRunStatus(
        error instanceof Error ? error.message : 'Failed to start sweep.',
      )
    } finally {
      setTuningSweepRunning(false)
    }
  }

  const handleRunLabelAudit = async () => {
    if (!selectedRun) return
    setLabelAuditStatus('')
    setLabelAuditRunning(true)
    setLabelAuditStatus('Queuing label audit...')
    try {
      const res = await fetchWithRetry(
        `${API_URL}/label-audits/run?run_id=${encodeURIComponent(
          selectedRun.run_id,
        )}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Label audit request failed (${res.status})`)
      }
      setLabelAuditStatus('Label audit queued. Refresh in a moment.')
      loadLabelAudits()
      window.setTimeout(() => loadLabelAudits(), 5000)
    } catch (error) {
      setLabelAuditStatus(
        error instanceof Error ? error.message : 'Failed to start label audit.',
      )
    } finally {
      setLabelAuditRunning(false)
    }
  }

  const handleRunDivergence = async () => {
    if (!selectedRun) return
    setDivergenceStatus('')
    setDivergenceRunning(true)
    setDivergenceStatus('Queuing divergence report...')
    try {
      const res = await fetchWithRetry(
        `${API_URL}/training-divergence/run?run_id=${encodeURIComponent(
          selectedRun.run_id,
        )}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Divergence request failed (${res.status})`)
      }
      setDivergenceStatus('Divergence report queued. Refresh in a moment.')
      loadDivergenceReports()
      window.setTimeout(() => loadDivergenceReports(), 5000)
    } catch (error) {
      setDivergenceStatus(
        error instanceof Error
          ? error.message
          : 'Failed to start divergence report.',
      )
    } finally {
      setDivergenceRunning(false)
    }
  }

  const isSubfolderActionRunning = (runId: string, action: string) =>
    subfolderActionKey === `${action}:${runId}`

  const handlePromoteRun = async (runId: string) => {
    setSubfolderStatus('')
    setSubfolderActionKey(`promote:${runId}`)
    try {
      const res = await fetchWithRetry(
        `${API_URL}/model-runs/${encodeURIComponent(runId)}/promote`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Promote request failed (${res.status})`)
      }
      const data = (await res.json()) as {
        promoted?: Array<{ status?: string }>
      }
      const promotedCount = (data.promoted || []).filter(
        (item) => item.status === 'copied',
      ).length
      setSubfolderStatus(
        promotedCount > 0
          ? `Promoted ${promotedCount} artifacts from ${runId}.`
          : `Promote complete for ${runId}.`,
      )
      await loadModelRuns({ preferredId: runId })
    } catch (error) {
      setSubfolderStatus(
        error instanceof Error ? error.message : 'Failed to promote run.',
      )
    } finally {
      setSubfolderActionKey('')
    }
  }

  const handleClearRun = async (runId: string) => {
    setSubfolderStatus('')
    setSubfolderActionKey(`clear:${runId}`)
    try {
      const res = await fetchWithRetry(
        `${API_URL}/model-runs/${encodeURIComponent(runId)}/clear`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Clear request failed (${res.status})`)
      }
      await res.json().catch(() => null)
      setSubfolderStatus(`Cleared run ${runId}. It can be retrained now.`)
      await loadModelRuns({ preferredId: runId })
    } catch (error) {
      setSubfolderStatus(
        error instanceof Error ? error.message : 'Failed to clear run.',
      )
    } finally {
      setSubfolderActionKey('')
    }
  }

  const handleOpenPredictForRun = (runId: string) => {
    const run = modelRuns.find((item) => item.run_id === runId)
    selectRun(runId, run)
    setActiveTab('predict')
  }

  const handleRunModelEvalForRun = async (runId: string) => {
    const run = modelRuns.find((item) => item.run_id === runId)
    selectRun(runId, run)
    setActiveTab('reports')
    setModelEvalRunning(true)
    setModelEvalRunStatus('Queuing evaluation run...')
    try {
      const res = await fetchWithRetry(
        `${API_URL}/model-eval/run?run_id=${encodeURIComponent(runId)}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        throw new Error(`Model eval request failed (${res.status})`)
      }
      const data = (await res.json()) as { report_id?: string }
      const reportId = data?.report_id
      setModelEvalRunStatus(
        reportId
          ? `Report queued: ${reportId}`
          : 'Report queued. Refresh to see status.',
      )
      loadModelEvalReports(reportId)
      if (reportId) {
        window.setTimeout(() => loadModelEvalReports(reportId), 5000)
      }
    } catch (error) {
      setModelEvalRunStatus('Failed to start model evaluation.')
    } finally {
      setModelEvalRunning(false)
    }
  }

  const handleRunLabelAuditForRun = async (runId: string) => {
    setLabelAuditStatus('')
    setLabelAuditRunning(true)
    setLabelAuditStatus('Queuing label audit...')
    try {
      const res = await fetchWithRetry(
        `${API_URL}/label-audits/run?run_id=${encodeURIComponent(runId)}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Label audit request failed (${res.status})`)
      }
      setLabelAuditStatus('Label audit queued. Refresh in a moment.')
      loadLabelAudits()
      window.setTimeout(() => loadLabelAudits(), 5000)
    } catch (error) {
      setLabelAuditStatus(
        error instanceof Error ? error.message : 'Failed to start label audit.',
      )
    } finally {
      setLabelAuditRunning(false)
    }
  }

  const handleRunDivergenceForRun = async (runId: string) => {
    setDivergenceStatus('')
    setDivergenceRunning(true)
    setDivergenceStatus('Queuing divergence report...')
    try {
      const res = await fetchWithRetry(
        `${API_URL}/training-divergence/run?run_id=${encodeURIComponent(runId)}`,
        { method: 'POST' },
      )
      if (!res.ok) {
        const payload = await res.json().catch(() => null)
        const detail = payload?.detail
        throw new Error(detail || `Divergence request failed (${res.status})`)
      }
      setDivergenceStatus('Divergence report queued. Refresh in a moment.')
      loadDivergenceReports()
      window.setTimeout(() => loadDivergenceReports(), 5000)
    } catch (error) {
      setDivergenceStatus(
        error instanceof Error
          ? error.message
          : 'Failed to start divergence report.',
      )
    } finally {
      setDivergenceRunning(false)
    }
  }

  const handleRunAuditForRun = async (runId: string) => {
    const run = modelRuns.find((item) => item.run_id === runId)
    selectRun(runId, run)
    setActiveTab('audit')
    await Promise.all([
      handleRunLabelAuditForRun(runId),
      handleRunDivergenceForRun(runId),
    ])
  }

  const selectedRun = useMemo(
    () => visibleModelRuns.find((run) => run.run_id === selectedRunId),
    [visibleModelRuns, selectedRunId],
  )

  const selectedIntegrityReport = useMemo(
    () =>
      integrityReports.find(
        (report) => report.report_id === selectedIntegrityId,
      ),
    [integrityReports, selectedIntegrityId],
  )

  const selectedLabelAudit = useMemo(
    () =>
      filteredLabelAudits.find(
        (report) => report.report_id === selectedLabelAuditId,
      ),
    [filteredLabelAudits, selectedLabelAuditId],
  )

  const selectedLabelAuditRun = useMemo(() => {
    if (!selectedLabelAudit?.run_id) return null
    return modelRuns.find((run) => run.run_id === selectedLabelAudit.run_id) || null
  }, [modelRuns, selectedLabelAudit])

  const selectedDivergenceReport = useMemo(
    () =>
      filteredDivergenceReports.find(
        (report) => report.report_id === selectedDivergenceId,
      ),
    [filteredDivergenceReports, selectedDivergenceId],
  )

  const selectedModelEvalReport = useMemo(
    () =>
      filteredModelEvalReports.find(
        (report) => report.report_id === selectedModelEvalId,
      ),
    [filteredModelEvalReports, selectedModelEvalId],
  )

  const selectedTuningSweep = useMemo(
    () =>
      tuningSweepReports.find(
        (report) => report.report_id === selectedTuningSweepId,
      ) || null,
    [tuningSweepReports, selectedTuningSweepId],
  )

  const filteredHistory = useMemo(() => {
    let rows = historyRows
    if (selectedLeague || selectedSeason) {
      rows = rows.filter((row) =>
        row.match_id ? matchById.has(row.match_id) : false,
      )
    }
    if (!historyFilter.trim()) return rows
    const needle = historyFilter.trim().toLowerCase()
    return rows.filter((row) => {
      const match = row.match_id ? matchById.get(row.match_id) : null
      const home = match?.home_team?.toLowerCase() || ''
      const away = match?.away_team?.toLowerCase() || ''
      return home.includes(needle) || away.includes(needle)
    })
  }, [historyRows, historyFilter, matchById, selectedLeague, selectedSeason])

  const matchesTsv = useMemo(
    () =>
      buildTsv(
        ['Date', 'Home', 'Away', 'Score'],
        matches.map((match) => [
          formatDate(match.date),
          match.home_team,
          match.away_team,
          formatScore(match),
        ]),
      ),
    [matches],
  )

  const predictionTsv = useMemo(() => {
    if (!predictionResult) return ''
    return buildTsv(
      ['Outcome', 'Probability'],
      [
        ['1 (Home)', formatProbability(predictionResult.win_prob_home)],
        ['X (Draw)', formatProbability(predictionResult.win_prob_draw)],
        ['2 (Away)', formatProbability(predictionResult.win_prob_away)],
      ],
    )
  }, [predictionResult])

  const historyTsv = useMemo(
    () =>
      buildTsv(
        ['Match', 'Model', 'Outcome', '1', 'X', '2'],
        filteredHistory.map((row) => {
          const match = row.match_id ? matchById.get(row.match_id) : null
          const predicted = labelFromProbs(
            row.win_prob_home,
            row.win_prob_draw,
            row.win_prob_away,
          )
          return [
            match
              ? `${match.home_team} vs ${match.away_team}`
              : row.match_id ?? 'Unknown',
            row.model_version,
            outcomeLabel(predicted),
            formatProbability(row.win_prob_home),
            formatProbability(row.win_prob_draw),
            formatProbability(row.win_prob_away),
          ]
        }),
      ),
    [filteredHistory, matchById],
  )

  const predictedLabel = useMemo(() => {
    if (!predictionResult) return null
    return labelFromProbs(
      predictionResult.win_prob_home,
      predictionResult.win_prob_draw,
      predictionResult.win_prob_away,
    )
  }, [predictionResult])

  const actualLabel = useMemo(() => outcomeFromScore(selectedMatch), [selectedMatch])

  const getMetricValue = (metrics: Record<string, unknown> | null, key: string) => {
    if (!metrics) return null
    const value = metrics[key]
    return typeof value === 'number' ? value : null
  }

  const getMetricArray = (metrics: Record<string, unknown> | null, key: string) => {
    if (!metrics) return null
    const value = metrics[key]
    return Array.isArray(value) ? value : null
  }

  const perClassTsv = useMemo(() => {
    const rows =
      (getMetricArray(
        selectedRun?.metrics || null,
        'test_f1_per_class',
      ) as Array<Record<string, unknown>> | null) || []
    return buildTsv(
      ['Label', 'Precision', 'Recall', 'F1', 'Support'],
      rows.map((row) => [
        String(row['label']),
        formatNumber(row['precision'] as number),
        formatNumber(row['recall'] as number),
        formatNumber(row['f1'] as number),
        row['support'] as number,
      ]),
    )
  }, [selectedRun])

  const artifactsTsv = useMemo(() => {
    const artifacts = selectedRun?.artifacts || []
    return buildTsv(
      ['Type', 'Path', 'Checksum', 'Size (bytes)'],
      artifacts.map((artifact) => [
        artifact.artifact_type,
        artifact.path,
        artifact.checksum,
        artifact.size_bytes,
      ]),
    )
  }, [selectedRun])

  const recentRuns = useMemo(
    () => visibleModelRuns.slice(0, runListLimit),
    [visibleModelRuns, runListLimit],
  )

  const visibleTopRuns = useMemo(() => {
    if (!flowFilterEnabled) return topRuns
    return topRuns.filter((run) => flowRunIds.has(run.run_id))
  }, [topRuns, flowFilterEnabled, flowRunIds])

  const bestRunIds = useMemo(() => {
    const ids = new Set<string>()
    for (const run of bestRuns) {
      if (run.run_id) {
        ids.add(run.run_id)
      }
    }
    return ids
  }, [bestRuns])

  const subfolderGroups = useMemo(() => {
    const grouped = new Map<string, ModelRun[]>()
    for (const run of modelRuns) {
      const key = run.model_name
      const existing = grouped.get(key)
      if (existing) {
        existing.push(run)
      } else {
        grouped.set(key, [run])
      }
    }
    return Array.from(grouped.entries())
      .map(([modelName, runs]) => ({
        modelName,
        runs: [...runs].sort(
          (a, b) => Date.parse(b.created_at) - Date.parse(a.created_at),
        ),
      }))
      .sort((a, b) => a.modelName.localeCompare(b.modelName))
  }, [modelRuns])

  const getDataValue = (
    data: Record<string, unknown> | null,
    key: string,
  ): string | number | null => {
    if (!data) return null
    const value = data[key]
    if (value == null) return null
    if (typeof value === 'string' || typeof value === 'number') return value
    if (typeof value === 'boolean') return value ? 'true' : 'false'
    return JSON.stringify(value)
  }

  const renderDateRange = (
    data: Record<string, unknown> | null,
    startKey: string,
    endKey: string,
  ) => {
    if (!data) return 'Not available'
    const startRaw = data[startKey]
    const endRaw = data[endKey]
    if (!startRaw && !endRaw) return 'Not available'
    const start = startRaw ? formatDate(String(startRaw)) : 'TBD'
    const end = endRaw ? formatDate(String(endRaw)) : 'TBD'
    return `${start} to ${end}`
  }

  const renderLeakageNotes = () => {
    if (!selectedRun?.data_summary) return 'Not available'
    const summary = selectedRun.data_summary
    const leakage = summary['leakage_checks'] as Record<string, unknown> | undefined
    if (!leakage) return 'Not available'
    const mismatch = leakage['mismatches']
    const dateOrder = leakage['date_order_violations']
    const negativeRest = leakage['negative_rest_days']
    return `Mismatches: ${mismatch ?? 0} | Date order: ${dateOrder ?? 0} | Negative rest days: ${negativeRest ?? 0}`
  }

  const renderCvSummary = () => {
    if (!selectedRun?.data_summary) return null
    const summary = selectedRun.data_summary['cv_summary'] as Record<string, unknown> | undefined
    if (!summary) return null
    const cvLogLoss = summary['cv_log_loss_mean']
    const cvBrier = summary['cv_brier_score_mean']
    const cvF1 = summary['cv_f1_macro_mean']
    const cvAcc = summary['cv_accuracy_mean']
    return `CV mean log loss: ${cvLogLoss ?? '-'} | CV mean brier: ${cvBrier ?? '-'} | CV mean F1: ${cvF1 ?? '-'} | CV mean acc: ${cvAcc ?? '-'}`
  }

  const clampFastIters = (value: number) =>
    Math.max(1, Math.min(2000, Math.floor(value)))

  const trainAllMainCommand = `docker compose exec -e PYTHONUNBUFFERED=1 api python -u -m app.ml.train_per_league --trainer main --model xgboost --continue-on-error --verbose-eval 1 --no-calibrate${
    forceRetrain ? ' --no-skip-existing' : ''
  }`
  const trainAllBaselineCommand = `docker compose exec -e PYTHONUNBUFFERED=1 api python -u -m app.ml.train_per_league --trainer baseline --continue-on-error --verbose-eval 1 --no-calibrate${
    forceRetrain ? ' --no-skip-existing' : ''
  }`
  const trainMainCommand = `docker compose exec api python -m app.ml.train_main --version v1 --model xgboost${
    selectedLeague ? ` --league ${selectedLeague}` : ''
  }${selectedSeason ? ` --season ${selectedSeason}` : ''}`
  const trainBaselineCommand = `docker compose exec api python -m app.ml.train_baseline --version v1${
    selectedLeague ? ` --league ${selectedLeague}` : ''
  }${selectedSeason ? ` --season ${selectedSeason}` : ''}`

  const isFlowTab = activeTab !== 'all'
  const flowIndex = isFlowTab ? FLOW_SEQUENCE.indexOf(activeTab as FlowTab) : -1
  const prevTab =
    isFlowTab && flowIndex > 0 ? FLOW_SEQUENCE[flowIndex - 1] : null
  const nextTab =
    isFlowTab && flowIndex < FLOW_SEQUENCE.length - 1
      ? FLOW_SEQUENCE[flowIndex + 1]
      : null
  const showAllSections = activeTab === 'all'
  const showTrain = showAllSections || activeTab === 'train'
  const showPredict = showAllSections || activeTab === 'predict'
  const showReports = showAllSections || activeTab === 'reports'
  const showAudit = showAllSections || activeTab === 'audit'

  const handlePredict = async () => {
    if (!selectedMatchId) return
    setPredictionLoading(true)
    setPredictionStatus('')
    setPredictionResult(null)
    try {
      const predictionModelName = selectedRun?.model_name || PREDICTION_MODEL
      const predictionFeatureVersion = selectedRun?.feature_version || 'v1'
      const predictionModelVersion = selectedRun?.model_version
      const params = new URLSearchParams({
        feature_version: predictionFeatureVersion,
        model_name: predictionModelName,
      })
      if (predictionModelVersion) {
        params.set('model_version', predictionModelVersion)
      }
      const response = await fetchWithRetry(
        `${API_URL}/predict?${params.toString()}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ match_id: Number(selectedMatchId) }),
        },
      )
      if (!response.ok) {
        throw new Error(`Prediction failed (${response.status})`)
      }
      const data = (await response.json()) as PredictionRecord
      setPredictionResult(data)
      setSelectedHistoryRecord(null)
      await handleLoadHistory()
      if (autoFlowNext === 'reports') {
        setActiveTab('reports')
        setAutoFlowNext('audit')
        setRunListMode('recent')
      }
    } catch (error) {
      setPredictionStatus('Failed to generate prediction.')
    } finally {
      setPredictionLoading(false)
    }
  }

  const handleLoadHistory = async () => {
    setHistoryLoading(true)
    setHistoryStatus('')
    try {
      const predictionModelName = selectedRun?.model_name || PREDICTION_MODEL
      const predictionModelVersion = selectedRun?.model_version
      const params = new URLSearchParams({
        limit: '200',
        model_name: predictionModelName,
      })
      if (predictionModelVersion) {
        params.set('model_version', predictionModelVersion)
      }
      const response = await fetchWithRetry(
        `${API_URL}/predictions?${params.toString()}`,
      )
      if (!response.ok) {
        throw new Error(`History failed (${response.status})`)
      }
      const data = await response.json()
      if (!Array.isArray(data)) {
        setHistoryRows([])
        return
      }
      const rows = data as PredictionRecord[]
      setHistoryRows(rows)
      setSelectedHistoryRecord((current) => {
        if (!current) return null
        const match = rows.find(
          (row) =>
            row.match_id === current.match_id &&
            row.model_version === current.model_version,
        )
        return match || null
      })
    } catch (error) {
      setHistoryStatus('Failed to load prediction history.')
    } finally {
      setHistoryLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="container py-4">
        <div className="header-card">
          <div>
            <p className="eyebrow">Football Data Viewer</p>
            <h1 className="display-5">Match results explorer</h1>
            <p className="lead text-muted">
              Select league and season to review sample matches seeded into Postgres.
            </p>
          </div>
          <div className={`status-badge status-${apiStatus}`}>
            API {apiStatus === 'checking' ? 'CHECK' : apiStatus.toUpperCase()}
          </div>
        </div>

        <div className="flow-tabs mt-3" role="tablist" data-testid="flow-tabs">
          {FLOW_TABS.map((tab) => {
            const flowNumber = FLOW_SEQUENCE.indexOf(tab.key as FlowTab)
            const displayIndex = flowNumber >= 0 ? flowNumber + 1 : 'All'
            return (
              <button
                key={tab.key}
                type="button"
                className={`flow-tab ${activeTab === tab.key ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.key)}
                role="tab"
                aria-selected={activeTab === tab.key}
                data-testid={`tab-${tab.key}`}
              >
                <span className="flow-index">{displayIndex}</span>
                <span className="flow-text">
                  <span className="flow-label">{tab.label}</span>
                  <span className="flow-caption">{tab.description}</span>
                </span>
              </button>
            )
          })}
        </div>

        {showTrain && (
          <>
            <section id="train" data-testid="section-train">
              <div className="card shadow-sm mt-4">
                <div className="card-body">
                  <div className="section-header">
                    <div>
                      <p className="eyebrow text-dark">Training</p>
                      <h2 className="section-title">Train all leagues</h2>
                      <p className="text-muted">
                        Run CLI training commands below. Use the data scope card to
                        target a specific league or season.
                      </p>
                    </div>
                    <div className="text-md-end">
                      <div className="scope-chip">
                        Scope: {formatLeagueLabel(selectedLeague)} |{' '}
                        {selectedSeason || 'All seasons'}
                      </div>
                      <div className="form-check form-switch mt-2 mb-0">
                        <input
                          className="form-check-input"
                          type="checkbox"
                          id="force-retrain"
                          checked={forceRetrain}
                          onChange={(event) => setForceRetrain(event.target.checked)}
                        />
                        <label className="form-check-label" htmlFor="force-retrain">
                          Force retrain (ignore existing)
                        </label>
                      </div>
                    </div>
                  </div>

                  <div className="command-grid mt-3">
                    <div className="command-card">
                      <div className="command-label">
                        All leagues - main model (XGBoost)
                      </div>
                      <pre className="command-block">{trainAllMainCommand}</pre>
                      <div className="d-flex align-items-center gap-2 flex-wrap mb-2">
                        <label className="form-label mb-0">Fast iters</label>
                        <input
                          type="number"
                          min={1}
                          max={2000}
                          step={1}
                          className="form-control form-control-sm"
                          style={{ maxWidth: 120 }}
                          value={fastTrainIters}
                          onChange={(event) => {
                            const next = Number(event.target.value)
                            if (!Number.isFinite(next)) return
                            setFastTrainIters(clampFastIters(next))
                          }}
                        />
                        <div className="small text-muted">e.g. 5, 10, 15</div>
                      </div>
                      <div className="d-flex gap-2 flex-wrap">
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-dark"
                          onClick={() =>
                            handleCopy(trainAllMainCommand, 'train-all-main')
                          }
                        >
                          {copyLabel('train-all-main', 'Copy command')}
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-dark"
                          onClick={() =>
                            handleRunTraining({
                              key: 'train-all-main-run',
                              label: 'all leagues (main model)',
                              trainer: 'main',
                              model: 'xgboost',
                            })
                          }
                          disabled={trainRunningKey !== ''}
                        >
                          {trainButtonLabel('train-all-main-run', 'Run training')}
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-secondary"
                          onClick={() =>
                            handleRunTraining({
                              key: 'train-all-main-fast',
                              label: `all leagues (main model, fast ${fastTrainIters} iters)`,
                              trainer: 'main',
                              model: 'xgboost',
                              overrides: {
                                n_estimators: String(clampFastIters(fastTrainIters)),
                              },
                              autoFlow: true,
                            })
                          }
                          disabled={trainRunningKey !== ''}
                        >
                          {trainButtonLabel(
                            'train-all-main-fast',
                            `Run fast (${fastTrainIters} iters)`,
                          )}
                        </button>
                      </div>
                    </div>
                    <div className="command-card">
                      <div className="command-label">
                        All leagues - baseline model
                      </div>
                      <pre className="command-block">{trainAllBaselineCommand}</pre>
                      <div className="d-flex gap-2 flex-wrap">
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-dark"
                          onClick={() =>
                            handleCopy(trainAllBaselineCommand, 'train-all-baseline')
                          }
                        >
                          {copyLabel('train-all-baseline', 'Copy command')}
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-dark"
                          onClick={() =>
                            handleRunTraining({
                              key: 'train-all-baseline-run',
                              label: 'all leagues (baseline model)',
                              trainer: 'baseline',
                              model: 'xgboost',
                            })
                          }
                          disabled={trainRunningKey !== ''}
                        >
                          {trainButtonLabel('train-all-baseline-run', 'Run training')}
                        </button>
                      </div>
                    </div>
                    <div className="command-card">
                      <div className="command-label">
                        Selected scope - main model
                      </div>
                      <div className="command-meta">
                        Uses: {formatLeagueLabel(selectedLeague)} |{' '}
                        {selectedSeason || 'All seasons'}
                      </div>
                      <pre className="command-block">{trainMainCommand}</pre>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-dark"
                        onClick={() => handleCopy(trainMainCommand, 'train-main-scope')}
                      >
                        {copyLabel('train-main-scope', 'Copy command')}
                      </button>
                    </div>
                    <div className="command-card">
                      <div className="command-label">
                        Selected scope - baseline model
                      </div>
                      <div className="command-meta">
                        Uses: {formatLeagueLabel(selectedLeague)} |{' '}
                        {selectedSeason || 'All seasons'}
                      </div>
                      <pre className="command-block">{trainBaselineCommand}</pre>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-dark"
                        onClick={() =>
                          handleCopy(trainBaselineCommand, 'train-baseline-scope')
                        }
                      >
                        {copyLabel('train-baseline-scope', 'Copy command')}
                      </button>
                    </div>
                  </div>

                  {trainStatus && (
                    <div className="alert alert-warning mt-3">{trainStatus}</div>
                  )}
                </div>
              </div>

              <div className="card shadow-sm mt-4">
                <div className="card-body">
                  <div className="section-header">
                    <div>
                      <p className="eyebrow text-dark">Model storage</p>
                      <h2 className="section-title">Model subfolders</h2>
                      <p className="text-muted">
                        The main `models/` folder keeps promoted artifacts. Each train
                        or fast-train run creates a subfolder under
                        `models/&lt;model_name&gt;/&lt;run_id&gt;`. Use the actions to
                        report, audit, predict, clear (allow rerun), or promote to the
                        main folder.
                      </p>
                    </div>
                    <div className="d-flex align-items-center gap-2 flex-wrap">
                      {(selectedLeague || selectedSeason) && (
                        <div className="small text-muted">
                          Scope: {formatLeagueLabel(selectedLeague)} |{' '}
                          {selectedSeason || 'All seasons'}
                        </div>
                      )}
                      {(selectedLeague || selectedSeason) && (
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-dark"
                          onClick={clearScope}
                        >
                          Clear scope
                        </button>
                      )}
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-dark"
                        onClick={() => loadModelRuns()}
                      >
                        Refresh
                      </button>
                    </div>
                  </div>

                  {subfolderStatus && (
                    <div className="alert alert-warning mt-3">{subfolderStatus}</div>
                  )}

                  {subfolderGroups.length === 0 && (
                    <p className="text-muted mt-3">
                      {selectedLeague || selectedSeason
                        ? 'No model subfolders found for the current scope.'
                        : 'No model subfolders found yet.'}
                    </p>
                  )}

                  {subfolderGroups.length > 0 && (
                    <div className="report-grid mt-3">
                      {subfolderGroups.map((group) => (
                        <div className="report-card" key={group.modelName}>
                          <h3 className="card-title">{group.modelName}</h3>
                          <div className="scroll-panel">
                            {group.runs.map((run, index) => (
                              <div
                                key={run.run_id}
                                className="run-tile has-actions"
                              >
                                <div className="run-info">
                                  <div className="run-title">
                                    Subfolder #{index + 1}
                                  </div>
                                  <div className="run-subtitle">
                                    {run.model_name}/{run.run_id}
                                  </div>
                                  <div className="run-meta">
                                    {formatDateTime(run.created_at)} |{' '}
                                    {formatLeagueLabel(run.league_code)}
                                    {run.season_label
                                      ? ` | ${run.season_label}`
                                      : ''}
                                  </div>
                                </div>
                                <div className="run-actions">
                                  <span
                                    className={`status-chip status-${run.status}`}
                                  >
                                    {run.status}
                                  </span>
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-dark"
                                    onClick={() => handleOpenPredictForRun(run.run_id)}
                                  >
                                    Predict
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-dark"
                                    onClick={() =>
                                      handleRunModelEvalForRun(run.run_id)
                                    }
                                    disabled={modelEvalRunning}
                                  >
                                    Report
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-dark"
                                    onClick={() => handleRunAuditForRun(run.run_id)}
                                    disabled={labelAuditRunning || divergenceRunning}
                                  >
                                    Audit
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-secondary"
                                    onClick={() => handlePromoteRun(run.run_id)}
                                    disabled={isSubfolderActionRunning(run.run_id, 'promote')}
                                  >
                                    {isSubfolderActionRunning(run.run_id, 'promote')
                                      ? 'Promoting...'
                                      : 'Promote'}
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-danger"
                                    onClick={() => handleClearRun(run.run_id)}
                                    disabled={isSubfolderActionRunning(run.run_id, 'clear')}
                                  >
                                    {isSubfolderActionRunning(run.run_id, 'clear')
                                      ? 'Clearing...'
                                      : 'Clear'}
                                  </button>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="card shadow-sm mt-4">
                <div className="card-body">
                  <div className="section-header">
                    <div>
                      <p className="eyebrow text-dark">Data scope</p>
                      <h2 className="section-title">League + season preview</h2>
                      <p className="text-muted">
                        Select a league and season to scope training, prediction, and
                        reporting.
                      </p>
                    </div>
                    <div className="text-md-end">
                      <div className="small text-muted" data-testid="matches-count">
                        Showing {matches.length} matches
                      </div>
                      <div className="small text-muted">{selectedLeagueName}</div>
                    </div>
                  </div>

                  <div className="row g-3 align-items-end mt-1">
                    <div className="col-md-5">
                      <label className="form-label">League</label>
                      <select
                        className="form-select"
                        value={selectedLeague}
                        onChange={(event) => setSelectedLeague(event.target.value)}
                        data-testid="league-select"
                      >
                        <option value="">Select league</option>
                        {leagues.map((league) => (
                          <option key={league.id} value={league.code}>
                            {league.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="col-md-4">
                      <label className="form-label">Season</label>
                      <select
                        className="form-select"
                        value={selectedSeason}
                        onChange={(event) => setSelectedSeason(event.target.value)}
                        disabled={!selectedLeague}
                        data-testid="season-select"
                      >
                        <option value="">All seasons</option>
                        {seasons.map((season) => (
                          <option key={season.id} value={season.label}>
                            {season.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {status && (
                    <div className="alert alert-warning mt-3">{status}</div>
                  )}

                  <div className="table-actions mt-3">
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => handleCopy(matchesTsv, 'matches-table')}
                      disabled={matches.length === 0}
                    >
                      {copyLabel('matches-table', 'Copy table')}
                    </button>
                    <span className="copy-hint">TSV for docs</span>
                  </div>

                  <div className="table-responsive mt-2">
                    <table
                      className="table table-striped align-middle"
                      data-testid="matches-table"
                    >
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Home</th>
                          <th>Away</th>
                          <th>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {matches.map((match) => (
                          <tr key={match.id}>
                            <td>{formatDate(match.date)}</td>
                            <td>{match.home_team}</td>
                            <td>{match.away_team}</td>
                            <td>{formatScore(match)}</td>
                          </tr>
                        ))}
                        {matches.length === 0 && (
                          <tr>
                            <td colSpan={4} className="text-muted text-center py-4">
                              {selectedLeague
                                ? 'No matches found for this scope.'
                                : 'Select a league to load matches.'}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
              </div>
            </div>
            </div>
            </section>

            {!showAllSections && (
              <div className="flow-footer">
                <div className="flow-path">
                  Train -&gt; Predict -&gt; Reports -&gt; Audit
                </div>
                <div className="flow-actions">
                  {prevTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => setActiveTab(prevTab)}
                    >
                      Back
                    </button>
                  )}
                  {nextTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-dark"
                      onClick={() => setActiveTab(nextTab)}
                    >
                      Next
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {showPredict && (
          <>
            <section id="predict" data-testid="section-predict">
          <div className="card shadow-sm mt-4">
            <div className="card-body">
              <div className="section-header">
                <div>
                  <p className="eyebrow text-dark">Predictions</p>
                  <h2 className="section-title">Match prediction</h2>
                  <p className="text-muted">
                    Generate a 1/X/2 prediction with probabilities using the
                    selected model run.
                  </p>
                </div>
                {renderFlowFilterControls('predict')}
              </div>

              <div className="row g-3 align-items-end">
                <div className="col-md-4">
                  <label className="form-label">Model run</label>
                  <select
                    className="form-select"
                    value={selectedRunId}
                    onChange={(event) => selectRun(event.target.value)}
                    disabled={visibleModelRuns.length === 0}
                  >
                    {visibleModelRuns.length === 0 && (
                      <option value="">No model runs found</option>
                    )}
                    {visibleModelRuns.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {run.model_name} | {run.model_version} |{' '}
                        {run.league_code || 'all'} {run.season_label || 'all'}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="col-md-5">
                  <label className="form-label">Match</label>
                  <select
                    className="form-select"
                    value={selectedMatchId}
                    onChange={(event) => setSelectedMatchId(event.target.value)}
                    disabled={matches.length === 0}
                    data-testid="prediction-match-select"
                  >
                    {matches.length === 0 && (
                      <option value="">
                        {selectedLeague
                          ? 'No matches found for this scope'
                          : 'Select league to load matches'}
                      </option>
                    )}
                    {matches.map((match) => (
                      <option key={match.id} value={String(match.id)}>
                        {match.home_team} vs {match.away_team} ({formatDate(match.date)})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="col-md-3 text-md-end">
                  <button
                    type="button"
                    className="btn btn-dark w-100"
                    onClick={handlePredict}
                    disabled={!selectedMatchId || predictionLoading}
                    data-testid="predict-button"
                  >
                    {predictionLoading ? 'Predicting…' : 'Predict'}
                  </button>
                </div>
              </div>

              {predictionStatus && (
                <div className="alert alert-warning mt-3">{predictionStatus}</div>
              )}

              {predictionResult && predictedLabel && (
                <div className="mt-3" data-testid="prediction-result">
                  <div className="d-flex flex-wrap align-items-center gap-3">
                    <div>
                      <div className="text-muted">Prediction</div>
                      <div className="h4 mb-0">
                        {outcomeLabel(predictedLabel)}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted">Actual</div>
                      <div className="h6 mb-0">
                        {actualLabel ? outcomeLabel(actualLabel) : 'TBD'}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted">Result</div>
                      <span
                        className={`status-chip ${
                          actualLabel
                            ? predictedLabel === actualLabel
                              ? ''
                              : 'status-failed'
                            : 'status-queued'
                        }`}
                      >
                        {actualLabel
                          ? predictedLabel === actualLabel
                            ? 'Correct'
                            : 'Incorrect'
                          : 'Pending'}
                      </span>
                    </div>
                    <div className="text-muted">
                      Model {predictionResult.model_name} ({predictionResult.model_version})
                    </div>
                    {selectedMatch && (
                      <div className="text-muted">
                        {selectedMatch.home_team} vs {selectedMatch.away_team} |{' '}
                        {formatLeagueLabel(selectedLeague)}{' '}
                        {selectedSeason ? `| ${selectedSeason}` : ''} |{' '}
                        {formatMatchDateTime(selectedMatch)} | Final:{' '}
                        {formatScore(selectedMatch)}
                      </div>
                    )}
                  </div>
                  <div className="table-actions mt-3">
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => handleCopy(predictionTsv, 'prediction-probs')}
                    >
                      {copyLabel('prediction-probs', 'Copy table')}
                    </button>
                    <span className="copy-hint">TSV for docs</span>
                  </div>
                  <div className="table-responsive mt-3">
                    <table className="table table-sm" data-testid="prediction-probs">
                      <thead>
                        <tr>
                          <th>Outcome</th>
                          <th>Probability</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>1 (Home)</td>
                          <td>{formatProbability(predictionResult.win_prob_home)}</td>
                        </tr>
                        <tr>
                          <td>X (Draw)</td>
                          <td>{formatProbability(predictionResult.win_prob_draw)}</td>
                        </tr>
                        <tr>
                          <td>2 (Away)</td>
                          <td>{formatProbability(predictionResult.win_prob_away)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="probability-chart" data-testid="prediction-chart">
                    <div className="probability-row">
                      <div className="probability-label">1</div>
                      <div className="probability-bar">
                        <div
                          className="probability-fill home"
                          style={{
                            width: `${
                              (predictionResult.win_prob_home ?? 0) * 100
                            }%`,
                          }}
                        />
                      </div>
                      <div className="probability-value">
                        {formatProbability(predictionResult.win_prob_home)}
                      </div>
                    </div>
                    <div className="probability-row">
                      <div className="probability-label">X</div>
                      <div className="probability-bar">
                        <div
                          className="probability-fill draw"
                          style={{
                            width: `${
                              (predictionResult.win_prob_draw ?? 0) * 100
                            }%`,
                          }}
                        />
                      </div>
                      <div className="probability-value">
                        {formatProbability(predictionResult.win_prob_draw)}
                      </div>
                    </div>
                    <div className="probability-row">
                      <div className="probability-label">2</div>
                      <div className="probability-bar">
                        <div
                          className="probability-fill away"
                          style={{
                            width: `${
                              (predictionResult.win_prob_away ?? 0) * 100
                            }%`,
                          }}
                        />
                      </div>
                      <div className="probability-value">
                        {formatProbability(predictionResult.win_prob_away)}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        <section id="history" data-testid="section-history">
          <div className="card shadow-sm mt-4">
            <div className="card-body">
              <div className="section-header">
                <div>
                  <p className="eyebrow text-dark">History</p>
                  <h2 className="section-title">Prediction history</h2>
                  <p className="text-muted">
                    Filter prediction history by team name.
                  </p>
                </div>
              </div>

              <div className="row g-3 align-items-end">
                <div className="col-md-6">
                  <label className="form-label">Filter by team</label>
                  <input
                    className="form-control"
                    value={historyFilter}
                    onChange={(event) => setHistoryFilter(event.target.value)}
                    placeholder="e.g. Arsenal"
                    data-testid="history-filter"
                  />
                </div>
                <div className="col-md-3">
                  <button
                    type="button"
                    className="btn btn-outline-dark w-100"
                    onClick={handleLoadHistory}
                    disabled={historyLoading}
                    data-testid="history-load"
                  >
                    {historyLoading ? 'Loading…' : 'Load history'}
                  </button>
                </div>
                <div className="col-md-3 text-md-end">
                  <div className="small text-muted">
                    League filter: {formatLeagueLabel(selectedLeague)}
                  </div>
                  <div className="small text-muted">
                    Season: {selectedSeason || 'All seasons'}
                  </div>
                </div>
              </div>

              {historyStatus && (
                <div className="alert alert-warning mt-3">{historyStatus}</div>
              )}

              <div className="table-actions mt-3">
                <button
                  type="button"
                  className="btn btn-sm btn-outline-dark"
                  onClick={() => handleCopy(historyTsv, 'history-table')}
                  disabled={filteredHistory.length === 0}
                >
                  {copyLabel('history-table', 'Copy table')}
                </button>
                <span className="copy-hint">TSV for docs</span>
              </div>

              <div className="table-responsive mt-3">
                <table className="table table-sm" data-testid="history-table">
                  <thead>
                    <tr>
                      <th>Match</th>
                      <th>Model</th>
                      <th>Outcome</th>
                      <th>1</th>
                      <th>X</th>
                      <th>2</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredHistory.map((row, index) => {
                      const match = row.match_id ? matchById.get(row.match_id) : null
                      const predicted = labelFromProbs(
                        row.win_prob_home,
                        row.win_prob_draw,
                        row.win_prob_away,
                      )
                      return (
                        <tr key={`${row.match_id}-${row.model_version}-${index}`}>
                          <td>
                            {match
                              ? `${match.home_team} vs ${match.away_team}`
                              : row.match_id ?? 'Unknown'}
                          </td>
                          <td>{row.model_version}</td>
                          <td>{outcomeLabel(predicted)}</td>
                          <td>{formatProbability(row.win_prob_home)}</td>
                          <td>{formatProbability(row.win_prob_draw)}</td>
                          <td>{formatProbability(row.win_prob_away)}</td>
                          <td>
                            <button
                              type="button"
                              className="btn btn-sm btn-outline-secondary"
                              onClick={() => setSelectedHistoryRecord(row)}
                              data-testid="history-detail-button"
                            >
                              Details
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                    {filteredHistory.length === 0 && (
                      <tr>
                        <td colSpan={7} className="text-muted text-center py-4">
                          {historyLoading
                            ? 'Loading…'
                            : 'No prediction history loaded.'}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {selectedHistoryRecord && (
                <div className="history-details" data-testid="history-details">
                  <div className="details-header">
                    <div>
                      <div className="text-muted">Selected record</div>
                      <div className="h6 mb-1">
                        {selectedHistoryRecord.match_id
                          ? `Match ${selectedHistoryRecord.match_id}`
                          : 'Unknown match'}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => setSelectedHistoryRecord(null)}
                    >
                      Close
                    </button>
                  </div>
                  <div className="row g-3 mt-1">
                    <div className="col-md-4">
                      <div className="key-label">Model version</div>
                      <div className="key-value">
                        {selectedHistoryRecord.model_version}
                      </div>
                    </div>
                    <div className="col-md-4">
                      <div className="key-label">Outcome</div>
                      <div className="key-value">
                        {outcomeLabel(
                          labelFromProbs(
                            selectedHistoryRecord.win_prob_home,
                            selectedHistoryRecord.win_prob_draw,
                            selectedHistoryRecord.win_prob_away,
                          ),
                        )}
                      </div>
                    </div>
                    <div className="col-md-4">
                      <div className="key-label">Probabilities</div>
                      <div className="key-value">
                        1 {formatProbability(selectedHistoryRecord.win_prob_home)} | X{' '}
                        {formatProbability(selectedHistoryRecord.win_prob_draw)} | 2{' '}
                        {formatProbability(selectedHistoryRecord.win_prob_away)}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

            {!showAllSections && (
              <div className="flow-footer">
                <div className="flow-path">
                  Train -&gt; Predict -&gt; Reports -&gt; Audit
                </div>
                <div className="flow-actions">
                  {prevTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => setActiveTab(prevTab)}
                    >
                      Back
                    </button>
                  )}
                  {nextTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-dark"
                      onClick={() => setActiveTab(nextTab)}
                    >
                      Next
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {showReports && (
          <>
            <section id="reports" data-testid="section-reports">
              <div className="card shadow-sm mt-4">
                <div className="card-body">
            <div className="section-header">
              <div>
                <p className="eyebrow text-dark">Model reports</p>
                <h2 className="section-title">Learned model intelligence</h2>
                <p className="text-muted">
                  Review model runs, training context, and evaluation metrics for the
                  selected league and season.
                </p>
              </div>
              <div className="report-controls">
                <div className="control-group">
                  <label className="form-label">View</label>
                  <select
                    className="form-select"
                    value={runListMode}
                    onChange={(event) =>
                      setRunListMode(event.target.value as RunListMode)
                    }
                  >
                    <option value="top">Top runs</option>
                    <option value="recent">Recent runs</option>
                  </select>
                </div>
                <div className="control-group">
                  <label className="form-label">Limit</label>
                  <select
                    className="form-select"
                    value={runListLimit}
                    onChange={(event) => setRunListLimit(Number(event.target.value))}
                  >
                    {RUN_LIMIT_OPTIONS.map((limit) => (
                      <option key={limit} value={limit}>
                        {limit}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="control-group">
                  <label className="form-label">Top metric</label>
                  <select
                    className="form-select"
                    value={reportMetric}
                    onChange={(event) => setReportMetric(event.target.value)}
                    disabled={runListMode !== 'top'}
                  >
                    {REPORT_METRICS.map((metric) => (
                      <option key={metric.key} value={metric.key}>
                        {metric.label}
                      </option>
                    ))}
                  </select>
                </div>
                {renderFlowFilterControls('reports', true)}
              </div>
            </div>

            {modelStatus && <div className="alert alert-warning mt-3">{modelStatus}</div>}

            <div className="report-grid report-grid--model mt-3">
              <div className="report-card">
                <h3 className="card-title">
                  {runListMode === 'top' ? 'Top model runs' : 'Recent model runs'}
                </h3>
                {runListMode === 'top' && visibleTopRuns.length === 0 && (
                  <p className="text-muted">No model runs found for this filter.</p>
                )}
                {runListMode === 'recent' && recentRuns.length === 0 && (
                  <p className="text-muted">No model runs found for this filter.</p>
                )}
                <div className="scroll-panel">
                  {runListMode === 'top' &&
                    visibleTopRuns.map((run, index) => (
                      <button
                        key={run.run_id}
                        type="button"
                        className={`run-tile ${
                          run.run_id === selectedRunId ? 'active' : ''
                        }`}
                        onClick={() => selectRun(run.run_id, run)}
                      >
                        <div>
                          <div className="run-title">
                            #{index + 1} {run.model_name}
                            {bestRunIds.has(run.run_id) && (
                              <span className="best-pill">Best</span>
                            )}
                          </div>
                          <div className="run-subtitle">
                            {formatLeagueLabel(run.league_code)} |{' '}
                            {run.season_label || 'All seasons'}
                          </div>
                          <div className="run-meta">
                            Version {run.model_version} | Feature {run.feature_version}
                          </div>
                        </div>
                        <div className="metric-pill">
                          {run.metric_name
                            ? run.metric_name.replace('test_', '')
                            : 'metric'}
                          <span>{formatNumber(run.metric)}</span>
                        </div>
                      </button>
                    ))}
                  {runListMode === 'recent' &&
                    recentRuns.map((run, index) => (
                      <button
                        key={run.run_id}
                        type="button"
                        className={`run-tile ${
                          run.run_id === selectedRunId ? 'active' : ''
                        }`}
                        onClick={() => selectRun(run.run_id, run)}
                      >
                        <div>
                          <div className="run-title">
                            #{index + 1} {run.model_name}
                            {bestRunIds.has(run.run_id) && (
                              <span className="best-pill">Best</span>
                            )}
                          </div>
                          <div className="run-subtitle">
                            {formatLeagueLabel(run.league_code)} |{' '}
                            {run.season_label || 'All seasons'}
                          </div>
                          <div className="run-meta">
                            {formatDateTime(run.created_at)} | Version{' '}
                            {run.model_version}
                          </div>
                        </div>
                        <div className="metric-pill">
                          {reportMetric.replace('test_', '')}
                          <span>
                            {formatNumber(getMetricValue(run.metrics, reportMetric))}
                          </span>
                        </div>
                      </button>
                    ))}
                </div>
              </div>

              <div className="report-card">
                <h3 className="card-title">Selected model overview</h3>
                {!selectedRun && (
                  <p className="text-muted">Pick a run to see its full report.</p>
                )}
                {selectedRun && (
                  <>
                    <div className="overview-header">
                      <div>
                        <div className="run-title">{selectedRun.model_name}</div>
                        <div className="run-subtitle">
                          {MODEL_DESCRIPTIONS[selectedRun.model_name] ||
                            'Custom training run for match outcomes.'}
                        </div>
                      </div>
                      <div className="run-badges">
                        <div className={`status-chip status-${selectedRun.status}`}>
                          {selectedRun.status}
                        </div>
                        {bestRunIds.has(selectedRun.run_id) && (
                          <div className="best-chip">Best</div>
                        )}
                      </div>
                    </div>

                    <div className="key-grid mt-3">
                      <div>
                        <div className="key-label">Run ID</div>
                        <div className="key-value">{selectedRun.run_id}</div>
                      </div>
                      <div>
                        <div className="key-label">Created</div>
                        <div className="key-value">
                          {formatDateTime(selectedRun.created_at)}
                        </div>
                      </div>
                      <div>
                        <div className="key-label">Model version</div>
                        <div className="key-value">{selectedRun.model_version}</div>
                      </div>
                      <div>
                        <div className="key-label">Feature version</div>
                        <div className="key-value">{selectedRun.feature_version}</div>
                      </div>
                      <div>
                        <div className="key-label">League</div>
                        <div className="key-value">
                          {formatLeagueLabel(selectedRun.league_code)}
                        </div>
                      </div>
                      <div>
                        <div className="key-label">Season</div>
                        <div className="key-value">
                          {selectedRun.season_label || 'All seasons'}
                        </div>
                      </div>
                    </div>

                  </>
                )}
              </div>

              <div className="report-card report-card--span">
                <h3 className="card-title">Run details</h3>
                {!selectedRun && (
                  <p className="text-muted">
                    Select a run to view training context, metrics, and reports.
                  </p>
                )}
                {selectedRun && (
                  <>
                    <div className="report-section">
                      <h4 className="section-subtitle">General training context</h4>
                      <div className="info-grid">
                        <div>
                          <div className="key-label">Train size</div>
                          <div className="key-value">
                            {getDataValue(selectedRun.data_summary, 'train_size') ?? '-'}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Validation size</div>
                          <div className="key-value">
                            {getDataValue(selectedRun.data_summary, 'val_size') ?? '-'}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Test size</div>
                          <div className="key-value">
                            {getDataValue(selectedRun.data_summary, 'test_size') ?? '-'}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Date range</div>
                          <div className="key-value">
                            {renderDateRange(
                              selectedRun.data_summary,
                              'min_date',
                              'max_date',
                            )}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Train period</div>
                          <div className="key-value">
                            {renderDateRange(
                              selectedRun.data_summary,
                              'train_min_date',
                              'train_max_date',
                            )}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Validation period</div>
                          <div className="key-value">
                            {renderDateRange(
                              selectedRun.data_summary,
                              'val_min_date',
                              'val_max_date',
                            )}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Test period</div>
                          <div className="key-value">
                            {renderDateRange(
                              selectedRun.data_summary,
                              'test_min_date',
                              'test_max_date',
                            )}
                          </div>
                        </div>
                        <div>
                          <div className="key-label">Leakage checks</div>
                          <div className="key-value">{renderLeakageNotes()}</div>
                        </div>
                        <div>
                          <div className="key-label">CV summary</div>
                          <div className="key-value">{renderCvSummary() || 'Not available'}</div>
                        </div>
                      </div>
                    </div>

                    <div className="report-section">
                      <h4 className="section-subtitle">Model metrics</h4>
                      <div className="metric-grid">
                        <div className="metric-card">
                          <div className="metric-label">Test log loss</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(selectedRun.metrics, 'test_log_loss'),
                            )}
                          </div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Test Brier score</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(selectedRun.metrics, 'test_brier_score'),
                            )}
                          </div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Test F1 macro</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(selectedRun.metrics, 'test_f1_macro'),
                            )}
                          </div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Test accuracy</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(selectedRun.metrics, 'test_accuracy'),
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="metric-grid mt-3">
                        <div className="metric-card">
                          <div className="metric-label">Baseline log loss (prior)</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(
                                selectedRun.metrics,
                                'test_baseline_prior_log_loss',
                              ),
                            )}
                          </div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Baseline Brier (prior)</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(
                                selectedRun.metrics,
                                'test_baseline_prior_brier_score',
                              ),
                            )}
                          </div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Baseline acc (majority)</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(
                                selectedRun.metrics,
                                'test_baseline_majority_accuracy',
                              ),
                            )}
                          </div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Baseline F1 (majority)</div>
                          <div className="metric-value">
                            {formatNumber(
                              getMetricValue(
                                selectedRun.metrics,
                                'test_baseline_majority_f1_macro',
                              ),
                            )}
                          </div>
                        </div>
                      </div>
                      {getMetricArray(selectedRun.metrics, 'test_f1_per_class') && (
                        <div className="mt-3">
                          <h5 className="section-subtitle">Per-class diagnostics</h5>
                          <div className="table-actions mt-2">
                            <button
                              type="button"
                              className="btn btn-sm btn-outline-dark"
                              onClick={() => handleCopy(perClassTsv, 'per-class')}
                            >
                              {copyLabel('per-class', 'Copy table')}
                            </button>
                            <span className="copy-hint">TSV for docs</span>
                          </div>
                          <div className="table-responsive">
                            <table className="table table-sm">
                              <thead>
                                <tr>
                                  <th>Label</th>
                                  <th>Precision</th>
                                  <th>Recall</th>
                                  <th>F1</th>
                                  <th>Support</th>
                                </tr>
                              </thead>
                              <tbody>
                                {(getMetricArray(
                                  selectedRun.metrics,
                                  'test_f1_per_class',
                                ) as Array<Record<string, unknown>>).map((row) => (
                                  <tr key={`${selectedRun.run_id}-${row['label']}`}>
                                    <td>{String(row['label'])}</td>
                                    <td>{formatNumber(row['precision'] as number)}</td>
                                    <td>{formatNumber(row['recall'] as number)}</td>
                                    <td>{formatNumber(row['f1'] as number)}</td>
                                    <td>{row['support'] as number}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                      <details className="details-block">
                        <summary>Full metrics & parameters</summary>
                        <div className="details-grid">
                          <div>
                            <div className="key-label">Metrics</div>
                            <pre className="json-block">
                              {JSON.stringify(selectedRun.metrics, null, 2)}
                            </pre>
                          </div>
                          <div>
                            <div className="key-label">Parameters</div>
                            <pre className="json-block">
                              {JSON.stringify(selectedRun.params, null, 2)}
                            </pre>
                          </div>
                        </div>
                      </details>
                    </div>

                    <div className="report-section">
                      <h4 className="section-subtitle">Artifacts</h4>
                      {selectedRun.artifacts && selectedRun.artifacts.length > 0 ? (
                        <>
                          <div className="table-actions mb-2">
                            <button
                              type="button"
                              className="btn btn-sm btn-outline-dark"
                              onClick={() => handleCopy(artifactsTsv, 'artifacts')}
                            >
                              {copyLabel('artifacts', 'Copy table')}
                            </button>
                            <span className="copy-hint">TSV for docs</span>
                          </div>
                          <div className="table-responsive">
                            <table className="table table-sm">
                              <thead>
                                <tr>
                                  <th>Type</th>
                                  <th>Path</th>
                                  <th>Checksum</th>
                                  <th>Size (bytes)</th>
                                </tr>
                              </thead>
                              <tbody>
                                {selectedRun.artifacts.map((artifact) => (
                                  <tr key={`${selectedRun.run_id}-${artifact.path}`}>
                                    <td>{artifact.artifact_type}</td>
                                    <td className="text-break">{artifact.path}</td>
                                    <td className="text-break">{artifact.checksum}</td>
                                    <td>{artifact.size_bytes}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </>
                      ) : (
                        <p className="text-muted">No artifacts logged for this run.</p>
                      )}
                    </div>

                    <div className="report-section">
                      <div className="section-header">
                        <div>
                          <h4 className="section-subtitle">Evaluation reports</h4>
                          <p className="text-muted">
                            Generate and review model vs baseline evaluation reports for
                            the selected run.
                          </p>
                        </div>
                        <div className="d-flex gap-2">
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-dark"
                            onClick={() => loadModelEvalReports(selectedModelEvalId)}
                          >
                            Refresh
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            onClick={handleRunModelEval}
                            disabled={!selectedRun || modelEvalRunning}
                          >
                            {modelEvalRunning ? 'Running...' : 'Run evaluation report'}
                          </button>
                        </div>
                      </div>

                      {modelEvalRunStatus && (
                        <div className="alert alert-info mt-2">
                          {modelEvalRunStatus}
                        </div>
                      )}
                      {modelEvalStatus && (
                        <div className="alert alert-warning mt-2">
                          {modelEvalStatus}
                        </div>
                      )}

                      <div className="report-grid mt-3">
                        <div className="report-card">
                          <h3 className="card-title">Available reports</h3>
                          {filteredModelEvalReports.length === 0 && (
                            <p className="text-muted">
                              No evaluation reports yet.
                            </p>
                          )}
                          <div className="scroll-panel">
                            {filteredModelEvalReports.map((report, index) => (
                              <button
                                key={report.report_id}
                                type="button"
                                className={`run-tile ${
                                  report.report_id === selectedModelEvalId
                                    ? 'active'
                                    : ''
                                }`}
                                onClick={() => setSelectedModelEvalId(report.report_id)}
                              >
                                <div>
                                  <div className="run-title">
                                    Report #{index + 1}{' '}
                                    {report.model_name ? `(${report.model_name})` : ''}
                                  </div>
                                  <div className="run-subtitle">
                                    Run {report.run_id || 'unknown'} |{' '}
                                    {report.model_version || 'n/a'}
                                  </div>
                                  <div className="run-meta">
                                    {formatDateTime(report.created_at)} |{' '}
                                    {formatLeagueLabel(report.league_code)}
                                    {report.season_label
                                      ? ` | ${report.season_label}`
                                      : ''}
                                  </div>
                                </div>
                                <div
                                  className={`status-chip status-${report.status}`}
                                >
                                  {report.status}
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="report-card">
                          <h3 className="card-title">Report details</h3>
                          {!selectedModelEvalId && (
                            <p className="text-muted">
                              Pick a report to see details.
                            </p>
                          )}
                          {selectedModelEvalId && (
                            <>
                              {selectedModelEvalReport?.status === 'running' && (
                                <div className="alert alert-info">
                                  Report is running. Refresh in a moment.
                                </div>
                              )}
                              {selectedModelEvalReport?.status === 'failed' &&
                                selectedModelEvalReport.error && (
                                  <div className="alert alert-warning">
                                    {selectedModelEvalReport.error}
                                  </div>
                                )}

                              <div className="integrity-actions">
                                {resolveReportUrl(
                                  selectedModelEvalReport?.metrics_json_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.metrics_json_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download metrics.json
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedModelEvalReport?.metrics_csv_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.metrics_csv_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download metrics.csv
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedModelEvalReport?.subsets_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.subsets_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download subsets.csv
                                  </a>
                                )}
                              </div>

                              {resolveReportUrl(
                                selectedModelEvalReport?.confusion_url,
                              ) && (
                                <div className="integrity-figure">
                                  <img
                                    src={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.confusion_url,
                                      ) || undefined
                                    }
                                    alt="Confusion matrix"
                                  />
                                </div>
                              )}

                              {resolveReportUrl(
                                selectedModelEvalReport?.confusion_baseline_url,
                              ) && (
                                <div className="integrity-figure">
                                  <img
                                    src={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.confusion_baseline_url,
                                      ) || undefined
                                    }
                                    alt="Baseline confusion matrix"
                                  />
                                </div>
                              )}

                              {resolveReportUrl(
                                selectedModelEvalReport?.calibration_url,
                              ) && (
                                <div className="integrity-figure">
                                  <img
                                    src={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.calibration_url,
                                      ) || undefined
                                    }
                                    alt="Calibration plot"
                                  />
                                </div>
                              )}

                              {resolveReportUrl(
                                selectedModelEvalReport?.entropy_hist_url,
                              ) && (
                                <div className="integrity-figure">
                                  <img
                                    src={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.entropy_hist_url,
                                      ) || undefined
                                    }
                                    alt="Entropy histogram"
                                  />
                                </div>
                              )}

                              {resolveReportUrl(
                                selectedModelEvalReport?.max_prob_hist_url,
                              ) && (
                                <div className="integrity-figure">
                                  <img
                                    src={
                                      resolveReportUrl(
                                        selectedModelEvalReport?.max_prob_hist_url,
                                      ) || undefined
                                    }
                                    alt="Max probability histogram"
                                  />
                                </div>
                              )}

                              <div className="integrity-markdown">
                                {modelEvalMarkdown ? (
                                  <pre>{modelEvalMarkdown}</pre>
                                ) : (
                                  <p className="text-muted">
                                    No summary loaded yet.
                                  </p>
                                )}
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="report-section">
                      <div className="section-header">
                        <div>
                          <h4 className="section-subtitle">Tuning sweeps</h4>
                          <p className="text-muted">
                            Run fast 300-iteration sweeps and export per-iteration
                            results for docs.
                          </p>
                        </div>
                        <div className="d-flex align-items-center gap-2 flex-wrap">
                          <div className="control-group">
                            <label className="form-label">Iterations</label>
                            <input
                              className="form-control form-control-sm"
                              type="number"
                              min={1}
                              value={tuningSweepIters}
                              onChange={(event) =>
                                setTuningSweepIters(
                                  Number(event.target.value) || 1,
                                )
                              }
                              style={{ maxWidth: 120 }}
                            />
                          </div>
                          <button
                            type="button"
                            className="btn btn-sm btn-outline-dark"
                            onClick={() =>
                              loadTuningSweepReports(selectedTuningSweepId)
                            }
                          >
                            Refresh
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm btn-primary"
                            onClick={handleRunTuningSweep}
                            disabled={tuningSweepRunning}
                          >
                            {tuningSweepRunning ? 'Running...' : 'Run sweep'}
                          </button>
                        </div>
                      </div>

                      {tuningSweepRunStatus && (
                        <div className="alert alert-info mt-2">
                          {tuningSweepRunStatus}
                        </div>
                      )}
                      {tuningSweepStatus && (
                        <div className="alert alert-warning mt-2">
                          {tuningSweepStatus}
                        </div>
                      )}

                      <div className="report-grid mt-3">
                        <div className="report-card">
                          <h3 className="card-title">Available sweeps</h3>
                          {tuningSweepReports.length === 0 && (
                            <p className="text-muted">No sweep runs yet.</p>
                          )}
                          <div className="scroll-panel">
                            {tuningSweepReports.map((report, index) => (
                              <button
                                key={report.report_id}
                                type="button"
                                className={`run-tile ${
                                  report.report_id === selectedTuningSweepId
                                    ? 'active'
                                    : ''
                                }`}
                                onClick={() =>
                                  setSelectedTuningSweepId(report.report_id)
                                }
                              >
                                <div>
                                  <div className="run-title">
                                    Sweep #{index + 1}{' '}
                                    {report.model ? `(${report.model})` : ''}
                                  </div>
                                  <div className="run-subtitle">
                                    {report.n_iter ?? 0} iters |{' '}
                                    {formatLeagueList(report.league_codes)}
                                  </div>
                                  <div className="run-meta">
                                    {formatDateTime(report.created_at)} |{' '}
                                    {report.season_label || 'All seasons'}
                                  </div>
                                  {report.status === 'running' && (
                                    <div className="run-meta">
                                      Progress:{' '}
                                      {report.percent != null
                                        ? `${report.percent.toFixed(1)}%`
                                        : '...'}{' '}
                                      {report.current_league
                                        ? `| ${report.current_league}`
                                        : ''}
                                      {report.current_iteration != null
                                        ? ` iter ${report.current_iteration}/${
                                            report.n_iter ?? '-'
                                          }`
                                        : ''}
                                    </div>
                                  )}
                                </div>
                                <div
                                  className={`status-chip status-${report.status}`}
                                >
                                  {report.status}
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="report-card">
                          <h3 className="card-title">Sweep details</h3>
                          {!selectedTuningSweepId && (
                            <p className="text-muted">
                              Pick a sweep to see details.
                            </p>
                          )}
                          {selectedTuningSweepId && (
                            <>
                              {selectedTuningSweep?.status === 'running' && (
                                <div className="alert alert-info">
                                  Sweep is running. Refresh in a moment.
                                </div>
                              )}
                              {selectedTuningSweep?.status === 'failed' &&
                                selectedTuningSweep.error && (
                                  <div className="alert alert-warning">
                                    {selectedTuningSweep.error}
                                  </div>
                                )}

                              {selectedTuningSweep?.status === 'running' && (
                                <div className="mt-3">
                                  <div className="metric-grid">
                                    <div className="metric-card">
                                      <div className="metric-label">Progress</div>
                                      <div className="metric-value">
                                        {selectedTuningSweep.percent != null
                                          ? `${selectedTuningSweep.percent.toFixed(
                                              1,
                                            )}%`
                                          : '-'}
                                      </div>
                                    </div>
                                    <div className="metric-card">
                                      <div className="metric-label">Current league</div>
                                      <div className="metric-value">
                                        {selectedTuningSweep.current_league || '-'}
                                      </div>
                                    </div>
                                    <div className="metric-card">
                                      <div className="metric-label">Iteration</div>
                                      <div className="metric-value">
                                        {selectedTuningSweep.current_iteration != null
                                          ? `${selectedTuningSweep.current_iteration}/${
                                              selectedTuningSweep.n_iter ?? '-'
                                            }`
                                          : '-'}
                                      </div>
                                    </div>
                                    <div className="metric-card">
                                      <div className="metric-label">Elapsed</div>
                                      <div className="metric-value">
                                        {formatDuration(
                                          selectedTuningSweep.elapsed_seconds,
                                        )}
                                      </div>
                                    </div>
                                    <div className="metric-card">
                                      <div className="metric-label">ETA</div>
                                      <div className="metric-value">
                                        {formatDuration(
                                          selectedTuningSweep.eta_seconds,
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                  {selectedTuningSweep.percent != null && (
                                    <div className="progress mt-3">
                                      <div
                                        className="progress-bar"
                                        role="progressbar"
                                        style={{
                                          width: `${Math.min(
                                            100,
                                            Math.max(
                                              0,
                                              selectedTuningSweep.percent,
                                            ),
                                          ).toFixed(1)}%`,
                                        }}
                                        aria-valuenow={Math.round(
                                          selectedTuningSweep.percent,
                                        )}
                                        aria-valuemin={0}
                                        aria-valuemax={100}
                                      />
                                    </div>
                                  )}
                                </div>
                              )}

                              {selectedTuningSweep && (
                                <div className="key-grid mt-3">
                                  <div>
                                    <div className="key-label">Report ID</div>
                                    <div className="key-value">
                                      {selectedTuningSweep.report_id}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="key-label">Created</div>
                                    <div className="key-value">
                                      {formatDateTime(
                                        selectedTuningSweep.created_at,
                                      )}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="key-label">Trainer</div>
                                    <div className="key-value">
                                      {selectedTuningSweep.trainer || '-'}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="key-label">Model</div>
                                    <div className="key-value">
                                      {selectedTuningSweep.model || '-'}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="key-label">Iterations</div>
                                    <div className="key-value">
                                      {selectedTuningSweep.n_iter ?? '-'}
                                    </div>
                                  </div>
                                  <div>
                                    <div className="key-label">Leagues</div>
                                    <div className="key-value">
                                      {formatLeagueList(
                                        selectedTuningSweep.league_codes,
                                      )}
                                    </div>
                                  </div>
                                </div>
                              )}

                              <div className="integrity-actions mt-3">
                                {resolveReportUrl(
                                  selectedTuningSweep?.results_csv_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.results_csv_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download results.csv
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedTuningSweep?.results_tsv_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.results_tsv_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download results.tsv
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedTuningSweep?.summary_by_league_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.summary_by_league_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download summary.csv
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedTuningSweep?.best_by_league_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.best_by_league_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download best_by_league.csv
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedTuningSweep?.results_md_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.results_md_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download markdown table
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedTuningSweep?.metadata_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.metadata_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download metadata.json
                                  </a>
                                )}
                                {resolveReportUrl(
                                  selectedTuningSweep?.errors_url,
                                ) && (
                                  <a
                                    className="btn btn-sm btn-outline-dark"
                                    href={
                                      resolveReportUrl(
                                        selectedTuningSweep?.errors_url,
                                      ) || undefined
                                    }
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    Download errors.csv
                                  </a>
                                )}
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
              </div>
              </div>
              </div>
            </section>

            {!showAllSections && (
              <div className="flow-footer">
                <div className="flow-path">
                  Train -&gt; Predict -&gt; Reports -&gt; Audit
                </div>
                <div className="flow-actions">
                  {prevTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => setActiveTab(prevTab)}
                    >
                      Back
                    </button>
                  )}
                  {nextTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-dark"
                      onClick={() => setActiveTab(nextTab)}
                    >
                      Next
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}

        {showAudit && (
          <>
            <div className="card shadow-sm mt-4">
          <div className="card-body">
            <div className="section-header">
              <div>
                <p className="eyebrow text-dark">Labels</p>
                <h2 className="section-title">Label audits</h2>
                <p className="text-muted">
                  Validate label encoding, confusion matrices, and prediction
                  distributions for recent training runs.
                </p>
              </div>
              <div className="d-flex align-items-center gap-3 flex-wrap">
                <button
                  type="button"
                  className="btn btn-sm btn-outline-dark"
                  onClick={() => loadLabelAudits(selectedLabelAuditId)}
                >
                  Refresh
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={handleRunLabelAudit}
                  disabled={!selectedRun || labelAuditRunning}
                >
                  {labelAuditRunning ? 'Generating...' : 'Generate for selected run'}
                </button>
                <div className="form-check form-switch mb-0">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id="label-audits-filter"
                    checked={filterAuditsToRun}
                    onChange={(event) => setFilterAuditsToRun(event.target.checked)}
                  />
                  <label className="form-check-label" htmlFor="label-audits-filter">
                    Selected run only
                  </label>
                </div>
                {renderFlowFilterControls('audit')}
              </div>
            </div>

            {labelAuditStatus && (
              <div className="alert alert-warning mt-3">{labelAuditStatus}</div>
            )}

            <div className="report-grid report-grid--stacked mt-3">
              <div className="report-card">
                <h3 className="card-title">Available audits</h3>
                {filteredLabelAudits.length === 0 && (
                  <p className="text-muted">No label audits found yet.</p>
                )}
                <div className="scroll-panel">
                  {filteredLabelAudits.map((report, index) => (
                    <button
                      key={report.report_id}
                      type="button"
                      className={`run-tile ${
                        report.report_id === selectedLabelAuditId ? 'active' : ''
                      }`}
                      onClick={() => setSelectedLabelAuditId(report.report_id)}
                    >
                      <div>
                        <div className="run-title">Audit #{index + 1}</div>
                        <div className="run-subtitle">Run {report.run_id}</div>
                        <div className="run-meta">
                          {formatDateTime(report.created_at)} |{' '}
                          {formatLeagueLabel(report.league_code)}
                          {report.season_label ? ` | ${report.season_label}` : ''}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="report-card">
                <h3 className="card-title">Audit summary</h3>
                {!selectedLabelAuditId && (
                  <p className="text-muted">Pick an audit to see the summary.</p>
                )}
                {selectedLabelAuditId && (
                  <>
                    {selectedLabelAuditRun ? (
                      <div className="metric-grid mt-2">
                        <div className="metric-card">
                          <div className="metric-label">Test accuracy</div>
                          <div className="metric-value">
                            {formatProbability(
                              getMetricValue(
                                selectedLabelAuditRun.metrics,
                                'test_accuracy',
                              ),
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-muted">No metrics available for this audit.</p>
                    )}
                  </>
                )}
              </div>

              <div className="report-card report-card--span">
                <h3 className="card-title">Audit details</h3>
                {!selectedLabelAuditId && (
                  <p className="text-muted">Pick an audit to see details.</p>
                )}
                {selectedLabelAuditId && (
                  <>
                    {resolveReportUrl(selectedLabelAudit?.chart_url) && (
                      <div className="integrity-figure">
                        <img
                          src={
                            resolveReportUrl(selectedLabelAudit?.chart_url) ||
                            undefined
                          }
                          alt="Predicted vs true distribution"
                        />
                      </div>
                    )}

                    {resolveReportUrl(selectedLabelAudit?.confusion_url) && (
                      <div className="integrity-figure">
                        <img
                          src={
                            resolveReportUrl(selectedLabelAudit?.confusion_url) ||
                            undefined
                          }
                          alt="Confusion matrix"
                        />
                      </div>
                    )}

                    <div className="integrity-markdown">
                      {labelAuditMarkdown ? (
                        <pre>{labelAuditMarkdown}</pre>
                      ) : (
                        <p className="text-muted">No summary loaded yet.</p>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="card shadow-sm mt-4">
          <div className="card-body">
            <div className="section-header">
              <div>
                <p className="eyebrow text-dark">Training stability</p>
                <h2 className="section-title">Training divergence monitor</h2>
                <p className="text-muted">
                  Track loss trends, learning rate, and scaling warnings to catch
                  unstable training runs.
                </p>
              </div>
              <div className="d-flex align-items-center gap-3 flex-wrap">
                <button
                  type="button"
                  className="btn btn-sm btn-outline-dark"
                  onClick={() => loadDivergenceReports(selectedDivergenceId)}
                >
                  Refresh
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={handleRunDivergence}
                  disabled={!selectedRun || divergenceRunning}
                >
                  {divergenceRunning
                    ? 'Generating...'
                    : 'Generate for selected run'}
                </button>
                <div className="form-check form-switch mb-0">
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id="divergence-filter"
                    checked={filterDivergenceToRun}
                    onChange={(event) =>
                      setFilterDivergenceToRun(event.target.checked)
                    }
                  />
                  <label className="form-check-label" htmlFor="divergence-filter">
                    Selected run only
                  </label>
                </div>
              </div>
            </div>

            {divergenceStatus && (
              <div className="alert alert-warning mt-3">{divergenceStatus}</div>
            )}

            <div className="report-grid report-grid--stacked mt-3">
              <div className="report-card">
                <h3 className="card-title">Available reports</h3>
                {filteredDivergenceReports.length === 0 && (
                  <p className="text-muted">No divergence reports found yet.</p>
                )}
                <div className="scroll-panel">
                  {filteredDivergenceReports.map((report, index) => (
                    <button
                      key={report.report_id}
                      type="button"
                      className={`run-tile ${
                        report.report_id === selectedDivergenceId ? 'active' : ''
                      }`}
                      onClick={() => setSelectedDivergenceId(report.report_id)}
                    >
                      <div>
                        <div className="run-title">Report #{index + 1}</div>
                        <div className="run-subtitle">Run {report.run_id}</div>
                        <div className="run-meta">
                          {formatDateTime(report.created_at)} |{' '}
                          {formatLeagueLabel(report.league_code)}
                          {report.season_label ? ` | ${report.season_label}` : ''}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="report-card">
                <h3 className="card-title">Report summary</h3>
                {!selectedDivergenceId && (
                  <p className="text-muted">Pick a report to see the summary.</p>
                )}
                {selectedDivergenceId && (
                  <div className="key-grid">
                    <div>
                      <div className="key-label">Run ID</div>
                      <div className="key-value">
                        {selectedDivergenceReport?.run_id || '-'}
                      </div>
                    </div>
                    <div>
                      <div className="key-label">Created</div>
                      <div className="key-value">
                        {formatDateTime(selectedDivergenceReport?.created_at)}
                      </div>
                    </div>
                    <div>
                      <div className="key-label">League</div>
                      <div className="key-value">
                        {formatLeagueLabel(selectedDivergenceReport?.league_code)}
                      </div>
                    </div>
                    <div>
                      <div className="key-label">Season</div>
                      <div className="key-value">
                        {selectedDivergenceReport?.season_label || 'All seasons'}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="report-card report-card--span">
                <h3 className="card-title">Report details</h3>
                {!selectedDivergenceId && (
                  <p className="text-muted">Pick a report to see details.</p>
                )}
                {selectedDivergenceId && (
                  <>
                    {resolveReportUrl(selectedDivergenceReport?.loss_chart_url) && (
                      <div className="integrity-figure">
                        <img
                          src={
                            resolveReportUrl(
                              selectedDivergenceReport?.loss_chart_url,
                            ) || undefined
                          }
                          alt="Loss and learning rate"
                        />
                      </div>
                    )}

                    {resolveReportUrl(selectedDivergenceReport?.grad_chart_url) && (
                      <div className="integrity-figure">
                        <img
                          src={
                            resolveReportUrl(
                              selectedDivergenceReport?.grad_chart_url,
                            ) || undefined
                          }
                          alt="Gradient norm"
                        />
                      </div>
                    )}

                    <div className="integrity-markdown">
                      {divergenceMarkdown ? (
                        <pre>{divergenceMarkdown}</pre>
                      ) : (
                        <p className="text-muted">No summary loaded yet.</p>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="card shadow-sm mt-4">
          <div className="card-body">
            <div className="section-header">
              <div>
                <p className="eyebrow text-dark">Data integrity</p>
                <h2 className="section-title">Dataset integrity audits</h2>
                <p className="text-muted">
                  Review alignment checks, missingness, and golden samples for the
                  latest ETL runs.
                </p>
              </div>
            </div>

            {integrityStatus && (
              <div className="alert alert-warning mt-3">{integrityStatus}</div>
            )}

            <div className="report-grid report-grid--stacked mt-3">
              <div className="report-card">
                <h3 className="card-title">Available audits</h3>
                {integrityReports.length === 0 && (
                  <p className="text-muted">No integrity reports found yet.</p>
                )}
                {integrityReports.map((report, index) => (
                  <button
                    key={report.report_id}
                    type="button"
                    className={`run-tile ${
                      report.report_id === selectedIntegrityId ? 'active' : ''
                    }`}
                    onClick={() => setSelectedIntegrityId(report.report_id)}
                  >
                    <div>
                      <div className="run-title">Audit #{index + 1}</div>
                      <div className="run-subtitle">{report.report_id}</div>
                      <div className="run-meta">
                        {formatDateTime(report.created_at)}
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              <div className="report-card">
                <h3 className="card-title">Audit summary</h3>
                {!selectedIntegrityId && (
                  <p className="text-muted">Pick an audit to see the summary.</p>
                )}
                {selectedIntegrityId && (
                  <>
                    <div className="integrity-actions">
                      {resolveReportUrl(selectedIntegrityReport?.golden_sample_url) && (
                        <a
                          className="btn btn-sm btn-outline-dark"
                          href={
                            resolveReportUrl(
                              selectedIntegrityReport?.golden_sample_url,
                            ) || undefined
                          }
                          target="_blank"
                          rel="noreferrer"
                        >
                          Download golden sample
                        </a>
                      )}
                    </div>
                    <div className="key-grid mt-3">
                      <div>
                        <div className="key-label">Report ID</div>
                        <div className="key-value">
                          {selectedIntegrityReport?.report_id || '-'}
                        </div>
                      </div>
                      <div>
                        <div className="key-label">Created</div>
                        <div className="key-value">
                          {formatDateTime(selectedIntegrityReport?.created_at)}
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>

              <div className="report-card report-card--span">
                <h3 className="card-title">Audit details</h3>
                {!selectedIntegrityId && (
                  <p className="text-muted">Pick an audit to see details.</p>
                )}
                {selectedIntegrityId && (
                  <>
                    {resolveReportUrl(selectedIntegrityReport?.missingness_url) && (
                      <div className="integrity-figure">
                        <img
                          src={
                            resolveReportUrl(
                              selectedIntegrityReport?.missingness_url,
                            ) || undefined
                          }
                          alt="Missingness chart"
                        />
                      </div>
                    )}

                    <div className="integrity-markdown">
                      {integrityMarkdown ? (
                        <pre>{integrityMarkdown}</pre>
                      ) : (
                        <p className="text-muted">No summary loaded yet.</p>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

            {!showAllSections && (
              <div className="flow-footer">
                <div className="flow-path">
                  Train -&gt; Predict -&gt; Reports -&gt; Audit
                </div>
                <div className="flow-actions">
                  {prevTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-dark"
                      onClick={() => setActiveTab(prevTab)}
                    >
                      Back
                    </button>
                  )}
                  {nextTab && (
                    <button
                      type="button"
                      className="btn btn-sm btn-dark"
                      onClick={() => setActiveTab(nextTab)}
                    >
                      Next
                    </button>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App

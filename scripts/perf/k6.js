import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const FEATURE_VERSION = __ENV.FEATURE_VERSION || 'v1';
const MODEL_NAME = __ENV.MODEL_NAME || 'main_xgboost';
const LEAGUE = __ENV.LEAGUE || '';
const SEASON = __ENV.SEASON || '';
const MATCH_LIMIT = parseInt(__ENV.MATCH_LIMIT || '200', 10);
const PREDICT_TIMEOUT = __ENV.PREDICT_TIMEOUT || '20s';

const parseIntEnv = (value, fallback) => {
  const parsed = parseInt(value ?? '', 10);
  return Number.isNaN(parsed) ? fallback : parsed;
};

const parseBoolEnv = (value, fallback = false) => {
  if (value == null) return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(value).toLowerCase());
};

const listMatchesVus = parseIntEnv(__ENV.LIST_MATCHES_VUS, 3);
const listPredictionsVus = parseIntEnv(__ENV.LIST_PREDICTIONS_VUS, 2);
const listDuration = __ENV.LIST_DURATION || '30s';

const steadyRate = parseIntEnv(__ENV.PREDICT_STEADY_RATE, 2);
const steadyDuration = __ENV.PREDICT_STEADY_DURATION || '60s';
const steadyPreAllocated = parseIntEnv(__ENV.PREDICT_STEADY_VUS, 10);
const steadyMaxVus = parseIntEnv(__ENV.PREDICT_STEADY_MAX_VUS, 30);

const burstEnabled = parseBoolEnv(__ENV.PREDICT_BURST, false);
const burstRate = parseIntEnv(__ENV.PREDICT_BURST_RATE, 8);
const burstDuration = __ENV.PREDICT_BURST_DURATION || '15s';
const burstPreAllocated = parseIntEnv(__ENV.PREDICT_BURST_VUS, 15);
const burstMaxVus = parseIntEnv(__ENV.PREDICT_BURST_MAX_VUS, 40);

const apiFailures = new Rate('api_failures');
const serverErrors = new Rate('server_errors');

function record(res) {
  const ok = res.status >= 200 && res.status < 300;
  apiFailures.add(!ok);
  serverErrors.add(res.status >= 500);
  return ok;
}

function pickRandom(items) {
  if (!items || items.length === 0) {
    return null;
  }
  const idx = Math.floor(Math.random() * items.length);
  return items[idx];
}

const scenarios = {
  list_matches: {
    executor: 'constant-vus',
    vus: listMatchesVus,
    duration: listDuration,
    exec: 'listMatches',
  },
  list_predictions: {
    executor: 'constant-vus',
    vus: listPredictionsVus,
    duration: listDuration,
    exec: 'listPredictions',
  },
  predict_steady: {
    executor: 'constant-arrival-rate',
    rate: steadyRate,
    timeUnit: '1s',
    duration: steadyDuration,
    preAllocatedVUs: steadyPreAllocated,
    maxVUs: steadyMaxVus,
    exec: 'predict',
  },
};

if (burstEnabled) {
  scenarios.predict_burst = {
    executor: 'constant-arrival-rate',
    rate: burstRate,
    timeUnit: '1s',
    duration: burstDuration,
    preAllocatedVUs: burstPreAllocated,
    maxVUs: burstMaxVus,
    exec: 'predict',
  };
}

export const options = {
  scenarios,
  thresholds: {
    http_req_failed: [__ENV.HTTP_REQ_FAILED_THRESHOLD || 'rate<0.05'],
    api_failures: [__ENV.API_FAILURE_THRESHOLD || 'rate<0.05'],
    server_errors: [__ENV.SERVER_ERROR_THRESHOLD || 'rate<0.01'],
    'http_req_duration{name:matches}': [
      __ENV.MATCHES_P95_THRESHOLD || 'p(95)<1500',
    ],
    'http_req_duration{name:predictions}': [
      __ENV.PREDICTIONS_P95_THRESHOLD || 'p(95)<1500',
    ],
    'http_req_duration{name:predict}': [
      __ENV.PREDICT_P95_THRESHOLD || 'p(95)<6000',
    ],
  },
  summaryTrendStats: ['avg', 'p(90)', 'p(95)', 'p(99)', 'min', 'max'],
};

export function setup() {
  const leaguesRes = http.get(`${BASE_URL}/leagues`, { tags: { name: 'leagues' } });
  if (!record(leaguesRes)) {
    throw new Error(`Failed to fetch leagues: ${leaguesRes.status}`);
  }
  const leagues = leaguesRes.json();
  const league = LEAGUE || (leagues.length ? leagues[0].code : null);
  if (!league) {
    throw new Error('No league available for performance tests.');
  }

  let season = SEASON;
  if (!season) {
    const seasonsRes = http.get(`${BASE_URL}/seasons?league=${encodeURIComponent(league)}`, {
      tags: { name: 'seasons' },
    });
    if (!record(seasonsRes)) {
      throw new Error(`Failed to fetch seasons: ${seasonsRes.status}`);
    }
    const seasons = seasonsRes.json();
    season = seasons.length ? seasons[0].label : null;
  }
  if (!season) {
    throw new Error('No season available for performance tests.');
  }

  const matchesUrl = `${BASE_URL}/matches?league=${encodeURIComponent(league)}&season=${encodeURIComponent(season)}&limit=${MATCH_LIMIT}`;
  const matchesRes = http.get(matchesUrl, { tags: { name: 'matches' } });
  if (!record(matchesRes)) {
    throw new Error(`Failed to fetch matches: ${matchesRes.status}`);
  }
  const matches = matchesRes.json();
  const matchIds = matches.map((m) => m.id).filter((id) => id !== null && id !== undefined);
  if (matchIds.length === 0) {
    throw new Error('No matches returned for performance tests.');
  }

  return {
    baseUrl: BASE_URL,
    league,
    season,
    matchIds,
    featureVersion: FEATURE_VERSION,
    modelName: MODEL_NAME,
  };
}

export function listMatches(data) {
  const url = `${data.baseUrl}/matches?league=${encodeURIComponent(data.league)}&season=${encodeURIComponent(data.season)}&limit=${MATCH_LIMIT}`;
  const res = http.get(url, { tags: { name: 'matches' } });
  check(res, { 'matches 200': (r) => r.status === 200 });
  record(res);
  sleep(0.2);
}

export function listPredictions(data) {
  const url = `${data.baseUrl}/predictions?model_name=${encodeURIComponent(data.modelName)}&limit=50`;
  const res = http.get(url, { tags: { name: 'predictions' } });
  check(res, { 'predictions 200': (r) => r.status === 200 });
  record(res);
  sleep(0.2);
}

export function predict(data) {
  const matchId = pickRandom(data.matchIds);
  if (matchId === null) {
    return;
  }
  const url = `${data.baseUrl}/predict?feature_version=${encodeURIComponent(data.featureVersion)}&model_name=${encodeURIComponent(data.modelName)}`;
  const payload = JSON.stringify({ match_id: matchId });
  const res = http.post(url, payload, {
    tags: { name: 'predict' },
    headers: { 'Content-Type': 'application/json' },
    timeout: PREDICT_TIMEOUT,
  });
  check(res, { 'predict 200': (r) => r.status === 200 });
  record(res);
}

#!/usr/bin/env python3

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

FIXTURES_DIR = Path(
    os.getenv("FIXTURES_DIR", "/tmp")
)

LEAGUES = {
    "premier-league": {
        "url": "https://www.soccerway.com/england/premier-league/fixtures/",
        "name": "English Premier League",
        "output": str(FIXTURES_DIR / "english-premier-league-fixtures.json"),
    },
    "championship": {
        "url": "https://www.soccerway.com/england/championship/fixtures/",
        "name": "English Championship",
        "output": str(FIXTURES_DIR / "english-championship-fixtures.json"),
    },
    "league-one": {
        "url": "https://www.soccerway.com/england/league-one/fixtures/",
        "name": "English League One",
        "output": str(FIXTURES_DIR / "english-league-one-fixtures.json"),
    },
    "league-two": {
        "url": "https://www.soccerway.com/england/league-two/fixtures/",
        "name": "English League Two",
        "output": str(FIXTURES_DIR / "english-league-two-fixtures.json"),
    },
    "national-league": {
        "url": "https://www.soccerway.com/england/national-league/fixtures/",
        "name": "English Conference",
        "output": str(FIXTURES_DIR / "english-conference-fixtures.json"),
    },
}


# Soccerway names -> names used by your football-predictor database.
# We'll add to this as we discover differences.
TEAM_NAMES = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester Utd": "Manchester United",
    "Nottingham": "Nott'm Forest",

    # Championship
    "Sheffield Utd": "Sheffield United",

    # League One
    "Bradford City": "Bradford",
    "Cambridge Utd": "Cambridge",
    "MK Dons": "Milton Keynes Dons",
    "Oxford Utd": "Oxford",
    "Peterborough": "Peterboro",
    "Sheffield Wed": "Sheffield Weds",
    "Stockport County": "Stockport",

    # League Two
    "Bristol Rovers": "Bristol Rvs",
    "Crawley": "Crawley Town",
    "Fleetwood": "Fleetwood Town",
    "Newport": "Newport County",
    "York City": "York",

    # National League
    "FC Halifax": "Halifax",
    "AFC Fylde": "Fylde",
}

def scrape_league(config):
    url = config["url"]

    print(f"\nDownloading {url}")

    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    scripts = soup.find_all("script")

    fixture_script = None

    for script in scripts:
        text = script.string or ""
        if "cjs.initialFeeds['fixtures']" in text:
            fixture_script = text
            break

    if fixture_script is None:
        raise RuntimeError(f"Could not find Soccerway fixtures feed for {url}")

    start = fixture_script.find("cjs.initialFeeds['fixtures']")
    data = fixture_script[start:]

    pattern = re.compile(
        r"AA÷([^¬]+)"
        r"¬AD÷([0-9]+)"
        r".*?"
        r"CX÷([^¬]+)"
        r"¬ER÷([^¬]+)"
        r".*?"
        r"AF÷([^¬]+)"
    )

    matches = pattern.findall(data)

    print(f"Found {len(matches)} fixtures")
    if not matches:
        raise RuntimeError(
            f"No fixtures found in Soccerway's embedded feed for {url}"
        )

    fixtures = []

    uk_timezone = ZoneInfo("Europe/London")
    today = datetime.now(uk_timezone).date()
    end_date = today + timedelta(days=14)

    for fixture_id, timestamp, home, round_name, away in matches:
        dt = datetime.fromtimestamp(
            int(timestamp),
            tz=ZoneInfo("UTC"),
        ).astimezone(uk_timezone)

        # Only keep fixtures from today through 14 days ahead.
        if not (today <= dt.date() <= end_date):
            continue

        home = TEAM_NAMES.get(home, home)
        away = TEAM_NAMES.get(away, away)

        fixtures.append({
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M"),
            "round": round_name,
            "team1": home,
            "team2": away,
        })

    fixtures.sort(key=lambda x: (x["date"], x["time"]))

    output = {
        "name": config["name"],
        "matches": fixtures,
    }

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"Written: {output_path}")
    print(f"Fixtures written: {len(fixtures)}")

    print("First 5 fixtures:")
    for fixture in fixtures[:5]:
        print(
            f"  {fixture['date']} {fixture['time']} | "
            f"{fixture['team1']} v {fixture['team2']} | "
            f"{fixture['round']}"
        )

    return fixtures


def main():
    total_fixtures = 0
    successful_leagues = 0
    failed_leagues = []

    for league_code, config in LEAGUES.items():
        try:
            fixtures = scrape_league(config)
            total_fixtures += len(fixtures)
            successful_leagues += 1

        except Exception as exc:
            failed_leagues.append((config["name"], str(exc)))
            print(
                f"\nWARNING: Failed to scrape {config['name']}: {exc}"
            )

    print("\nScraping summary")
    print(f"Leagues processed successfully: {successful_leagues}")
    print(f"Leagues failed: {len(failed_leagues)}")
    print(f"Total fixtures found: {total_fixtures}")

    if failed_leagues:
        print("\nFailed leagues:")
        for league_name, error in failed_leagues:
            print(f"  {league_name}: {error}")
        raise RuntimeError(
            f"Fixture scraping failed for {len(failed_leagues)} league(s)"
        )

if __name__ == "__main__":
    main()
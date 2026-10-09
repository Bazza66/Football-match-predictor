#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Project locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SERVER = PROJECT_ROOT / "python-server"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

SCRAPER = PROJECT_ROOT / "scripts" / "soccerway_fixtures.py"
FEATURE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_season_features.py"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_weekly_predictions.py"

FIXTURES_DIR = Path(os.getenv("FIXTURES_DIR", "/tmp"))
SEASON_LABEL = os.getenv("SEASON_LABEL", "2026-27")
TEMP_DATA = FIXTURES_DIR / "football-fixture-update" / SEASON_LABEL

LEAGUE_FILES = [
    ("english-premier-league-fixtures.json", "english-premier-league.json"),
    ("english-championship-fixtures.json", "english-championship.json"),
    ("english-league-one-fixtures.json", "english-league-one.json"),
    ("english-league-two-fixtures.json", "english-league-two.json"),
    ("english-conference-fixtures.json", "english-conference.json"),
]


def run(command, cwd=None):
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("=" * 80)

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def main():
    print("Football prediction update")
    print(f"Project: {PROJECT_ROOT}")

    if not VENV_PYTHON.exists():
        raise RuntimeError(
            f"Python virtual environment not found: {VENV_PYTHON}"
        )

    if not PYTHON_SERVER.is_dir():
        raise RuntimeError(
            f"Python server directory not found: {PYTHON_SERVER}"
        )

    for script in (SCRAPER, FEATURE_SCRIPT, EXPORT_SCRIPT):
        if not script.is_file():
            raise RuntimeError(f"Required script not found: {script}")

    # ------------------------------------------------------------------
    # 1. Scrape Soccerway
    # ------------------------------------------------------------------
    print("\nSTEP 1: Scraping Soccerway fixtures")

    run(
        [str(VENV_PYTHON), str(SCRAPER)],
        cwd=PROJECT_ROOT,
    )

    # ------------------------------------------------------------------
    # 2. Prepare temporary ETL data directory
    # ------------------------------------------------------------------
    print("\nSTEP 2: Preparing fixture data")

    TEMP_DATA.mkdir(parents=True, exist_ok=True)

    for source_name, destination_name in LEAGUE_FILES:
        source = FIXTURES_DIR / source_name
        destination = TEMP_DATA / destination_name

        if not source.is_file():
            raise RuntimeError(
                f"Expected scraper output not found: {source}"
            )

        shutil.copy2(source, destination)
        print(f"  {source.name} -> {destination}")

    # ------------------------------------------------------------------
    # 3. Import/update matches
    # ------------------------------------------------------------------
    print("\nSTEP 3: Importing fixtures into database")

    run(
        [
            str(VENV_PYTHON),
            "-m",
            "app.etl.load_football_json",
            "--data-dir",
            str(TEMP_DATA.parent),
            "--no-features",
        ],
        cwd=PYTHON_SERVER,
    )

    # ------------------------------------------------------------------
    # 4. Generate features
    # ------------------------------------------------------------------
    print("\nSTEP 4: Generating features")

    run(
        [str(VENV_PYTHON), str(FEATURE_SCRIPT)],
        cwd=PYTHON_SERVER,
    )

    # ------------------------------------------------------------------
    # 5. Generate predictions for anything not already predicted
    # ------------------------------------------------------------------
    print("\nSTEP 5: Generating missing predictions")

    run(
        [
            str(VENV_PYTHON),
            "-m",
            "app.ml.predict_to_db",
            "--version",
            "v1",
            "--model",
            "main_xgboost",
        ],
        cwd=PYTHON_SERVER,
    )

    # ------------------------------------------------------------------
    # 6. Export current week's predictions to CSV
    # ------------------------------------------------------------------
    print("\nSTEP 6: Exporting current week's predictions")

    run(
        [str(VENV_PYTHON), str(EXPORT_SCRIPT)],
        cwd=PYTHON_SERVER,
    )

    print("\n" + "=" * 80)
    print("UPDATE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: Command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

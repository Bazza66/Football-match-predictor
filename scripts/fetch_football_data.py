#!/usr/bin/env python3

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


# Default to the repository's football.json-master directory.
# DATA_DIR can override this when a different output location is required.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(
    os.getenv("DATA_DIR", str(PROJECT_ROOT / "football.json-master"))
)

LEAGUES = {
    "E0": "english-premier-league",
    "E1": "english-championship",
    "E2": "english-league-one",
    "E3": "english-league-two",
    "E4": "english-conference",
}

START_YEAR = 1993
END_YEAR = 2027  # Exclusive: includes the 2026-27 season.

DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d")


def read_csv_with_fallback(url):
    """Read a CSV using the default encoding, then UTF-8 and CP1252."""
    encodings = (None, "utf-8", "cp1252")

    for encoding in encodings:
        try:
            options = {"on_bad_lines": "skip"}
            if encoding is not None:
                options["encoding"] = encoding

            df = pd.read_csv(url, **options)
            return df, encoding or "pandas default"

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Could not decode {url} using default, UTF-8 or CP1252"
    )


def parse_date(value):
    """Parse a match date using the supported date formats."""
    date_str = str(value).strip()

    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue

    return None


def convert_matches(df):
    """Convert a football-data.co.uk dataframe to the project's JSON format."""
    required_columns = {"Date", "HomeTeam", "AwayTeam"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    # Keep unplayed matches, but discard rows without date or team names.
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    matches = []
    skipped_rows = 0

    for _, row in df.iterrows():
        try:
            match_date = parse_date(row["Date"])

            if match_date is None:
                skipped_rows += 1
                continue

            home_goals = row.get("FTHG")
            away_goals = row.get("FTAG")

            if pd.notna(home_goals) and pd.notna(away_goals):
                score = {"ft": [int(home_goals), int(away_goals)]}
            else:
                score = {"ft": None}

            matches.append(
                {
                    "date": match_date.strftime("%Y-%m-%d"),
                    "team1": str(row["HomeTeam"]).strip(),
                    "team2": str(row["AwayTeam"]).strip(),
                    "score": score,
                }
            )

        except (ValueError, TypeError, KeyError) as exc:
            skipped_rows += 1
            print(f"    Warning: skipped a row: {exc}")

    return matches, skipped_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_matches = 0
    total_skipped = 0
    successful_files = 0
    failed_files = 0

    print("Football-data.co.uk historical data download")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Season start years: {START_YEAR} to {END_YEAR - 1}")
    print()

    for year in range(START_YEAR, END_YEAR):
        season_code = f"{year % 100:02d}{(year + 1) % 100:02d}"
        season_name = f"{year}-{(year + 1) % 100:02d}"
        season_folder = OUTPUT_DIR / season_name
        season_folder.mkdir(parents=True, exist_ok=True)

        for league_code, league_name in LEAGUES.items():
            url = (
                "https://football-data.co.uk/mmz4281/"
                f"{season_code}/{league_code}.csv"
            )
            print(f"Downloading {league_name}, season {season_name}...")

            try:
                df, encoding_used = read_csv_with_fallback(url)
                matches, skipped_rows = convert_matches(df)

                output_file = season_folder / f"{league_name}.json"
                output_data = {
                    "name": league_name.replace("-", " ").title(),
                    "matches": matches,
                }

                with output_file.open("w", encoding="utf-8") as file:
                    json.dump(output_data, file, indent=2, ensure_ascii=False)

                total_matches += len(matches)
                total_skipped += skipped_rows
                successful_files += 1

                print(
                    f"    Saved {len(matches)} matches; "
                    f"{skipped_rows} rows skipped; "
                    f"encoding: {encoding_used}"
                )

            except Exception as exc:
                failed_files += 1
                print(f"    WARNING: could not process {url}: {exc}")

    print("\nDownload summary")
    print(f"Files processed successfully: {successful_files}")
    print(f"Files unavailable or failed: {failed_files}")
    print(f"Matches saved: {total_matches}")
    print(f"Rows skipped during conversion: {total_skipped}")
    if failed_files:
        raise RuntimeError(
            f"Failed to process {failed_files} file(s)"
    )

    print("Download finished successfully.")
 
if __name__ == "__main__":
    main()
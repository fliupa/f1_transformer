"""Fetch missing F1 data from FastF1 API with rate limit handling.

Strategy:
1. Skip fully complete years
2. Add delays between sessions to stay under 500 calls/hour
3. Save incrementally per year
4. Merge all data at the end
"""
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
F1_CACHE = PROJECT_ROOT / "data" / "f1_cache"
RAW_DATA.mkdir(parents=True, exist_ok=True)
(RAW_DATA / "races").mkdir(exist_ok=True)

import fastf1
fastf1.Cache.enable_cache(str(F1_CACHE))

# Complete seasons (all races present)
# Check existing year files
EXISTING_YEARS = {}
for y in range(2014, 2026):
    fpath = RAW_DATA / "races" / f"race_results_{y}.csv"
    if fpath.exists():
        df = pd.read_csv(fpath)
        EXISTING_YEARS[y] = len(df)

print("Existing data:")
for y, n in sorted(EXISTING_YEARS.items()):
    print(f"  {y}: {n} entries")

# Expected races per year (F1 seasons)
EXPECTED_RACES = {
    2014: 19, 2015: 19, 2016: 21, 2017: 20, 2018: 21, 2019: 21,
    2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24, 2025: 24,
}

# Determine which years to fetch
YEARS_TO_FETCH = []
for y in range(2014, 2026):
    expected_drivers = EXPECTED_RACES.get(y, 20) * 20  # rough estimate
    existing = EXISTING_YEARS.get(y, 0)
    # A full season has ~20-24 races * ~20 drivers = ~400-480 entries
    # If we have more than 85% of expected, consider it complete
    threshold = expected_drivers * 0.85
    if existing < threshold:
        YEARS_TO_FETCH.append(y)
        print(f"  Year {y}: {existing} entries < {int(threshold)} threshold, WILL FETCH")
    else:
        print(f"  Year {y}: {existing} entries, SKIP")

print(f"\nYears to fetch: {YEARS_TO_FETCH}")
print(f"Total years: {len(YEARS_TO_FETCH)}")


def _safe_td_seconds(td):
    """Safely convert timedelta to seconds."""
    if td is None:
        return np.nan
    try:
        if hasattr(td, 'total_seconds'):
            return td.total_seconds()
        if pd.isna(td):
            return np.nan
        return float(td)
    except Exception:
        return np.nan


def fetch_season_results(year: int) -> list[dict]:
    """Fetch race results for an entire season."""
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"  Could not get schedule for {year}: {e}")
        return []

    results_list = []
    events = schedule[schedule.EventName.notna() & (schedule.EventFormat != "test")]

    for _, event in events.iterrows():
        event_name = event["EventName"]
        country = event.get("Country", "Unknown")
        round_num = int(event.get("RoundNumber", 0))
        if round_num == 0:
            continue
        try:
            event_date = pd.Timestamp(event["EventDate"])
        except Exception:
            event_date = pd.NaT

        # Rate limit: delay between races to stay under 500 calls/hour
        time.sleep(1.0)

        # Get race session
        race = None
        try:
            race = fastf1.get_session(year, event_name, "R")
            race.load(telemetry=False, weather=False, messages=False)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                print(f"    RATE LIMITED! Waiting 60s...")
                time.sleep(60)
                try:
                    race = fastf1.get_session(year, event_name, "R")
                    race.load(telemetry=False, weather=False, messages=False)
                except Exception as e2:
                    print(f"    Still failed: {e2}")
                    continue
            else:
                print(f"    Skipping race {year} {event_name}: {e}")
                continue

        if race is None or not hasattr(race, "results") or race.results is None:
            continue

        # Get qualifying session
        quali_data = {}
        try:
            time.sleep(0.2)
            quali = fastf1.get_session(year, event_name, "Q")
            quali.load(telemetry=False, weather=False, messages=False)
            quali_results = quali.results
            for _, drv in quali_results.iterrows():
                key = drv.get("Abbreviation", drv.get("DriverNumber"))
                quali_data[key] = {
                    "q_position": drv.get("Position", np.nan),
                    "q1_time": _safe_td_seconds(drv.get("Q1")),
                    "q2_time": _safe_td_seconds(drv.get("Q2")),
                    "q3_time": _safe_td_seconds(drv.get("Q3")),
                }
        except Exception:
            pass

        # Get sprint results if available
        sprint_data = {}
        try:
            time.sleep(0.15)
            sprint = fastf1.get_session(year, event_name, "S")
            sprint.load(telemetry=False, weather=False, messages=False)
            for _, drv in sprint.results.iterrows():
                key = drv.get("Abbreviation", drv.get("DriverNumber"))
                sprint_data[key] = {
                    "sprint_position": drv.get("Position", np.nan),
                    "sprint_points": drv.get("Points", 0),
                }
        except Exception:
            pass

        # Circuit name
        circuit_name = str(event_name)
        try:
            if hasattr(race, "event"):
                ev = race.event
                if isinstance(ev, dict):
                    circuit_name = ev.get("CircuitName", event_name)
                elif hasattr(ev, "get"):
                    circuit_name = ev.get("CircuitName", event_name)
                else:
                    circuit_name = str(ev) if ev else str(event_name)
        except Exception:
            pass

        # Weather data
        weather_avg = {}
        try:
            weather = race.weather_data
            if weather is not None and not weather.empty:
                weather_avg = {
                    "air_temp": weather.get("AirTemp", pd.Series()).mean() if "AirTemp" in weather.columns else np.nan,
                    "track_temp": weather.get("TrackTemp", pd.Series()).mean() if "TrackTemp" in weather.columns else np.nan,
                    "humidity": weather.get("Humidity", pd.Series()).mean() if "Humidity" in weather.columns else np.nan,
                    "pressure": weather.get("Pressure", pd.Series()).mean() if "Pressure" in weather.columns else np.nan,
                    "wind_speed": weather.get("WindSpeed", pd.Series()).mean() if "WindSpeed" in weather.columns else np.nan,
                    "rainfall": weather.get("Rainfall", pd.Series()).sum() if "Rainfall" in weather.columns else 0.0,
                }
                weather_avg["rain_flag"] = 1 if weather_avg.get("rainfall", 0) > 0 else 0
        except Exception:
            pass

        for _, drv in race.results.iterrows():
            abbr = drv.get("Abbreviation", str(drv.get("DriverNumber", "?")))
            driver_num = drv.get("DriverNumber", 0)
            grid_pos = drv.get("GridPosition", 20)
            finish_pos = drv.get("Position", 20)
            status = str(drv.get("Status", "Finished"))
            dnf = 0 if "Finished" in status or "+" in status or "Lap" in status else 1

            q_info = quali_data.get(abbr, quali_data.get(driver_num, {}))
            s_info = sprint_data.get(abbr, sprint_data.get(driver_num, {}))

            row = {
                "year": year,
                "round": round_num,
                "event_name": event_name,
                "country": country,
                "event_date": event_date,
                "circuit": circuit_name,
                "driver_abbreviation": abbr,
                "driver_number": driver_num,
                "constructor": drv.get("TeamName", "Unknown"),
                "grid_position": grid_pos if grid_pos and not pd.isna(grid_pos) else 20,
                "finish_position": finish_pos if finish_pos and not pd.isna(finish_pos) else 20,
                "points": drv.get("Points", 0),
                "dnf": dnf,
                "status": status,
                **weather_avg,
                "q_position": q_info.get("q_position", np.nan),
                "q1_time": q_info.get("q1_time", np.nan),
                "q2_time": q_info.get("q2_time", np.nan),
                "q3_time": q_info.get("q3_time", np.nan),
                "sprint_position": s_info.get("sprint_position", np.nan),
                "sprint_points": s_info.get("sprint_points", 0),
            }
            results_list.append(row)

        print(f"  [{year}] R{int(round_num):2d}: {event_name:<25s} - {len(race.results)} drivers")

    return results_list


def main():
    print("=" * 60)
    print("FETCHING MISSING F1 DATA FROM FastF1")
    print(f"Years: {YEARS_TO_FETCH}")
    print("=" * 60)

    all_new_data = []

    for year in YEARS_TO_FETCH:
        print(f"\nSeason {year}...")
        season = fetch_season_results(year)

        if season:
            # Save per-year file
            year_df = pd.DataFrame(season)
            save_path = RAW_DATA / "races" / f"race_results_{year}.csv"
            year_df.to_csv(save_path, index=False)
            print(f"  Saved {len(season)} entries ({year_df['round'].nunique()} races) to {save_path}")
            all_new_data.extend(season)
        else:
            print(f"  No data fetched for {year}")

        # Delay between seasons to avoid rate limits (especially after a full season fetch)
        if year != YEARS_TO_FETCH[-1]:
            wait_time = 15
            print(f"  Pausing {wait_time}s before next season to avoid rate limits...")
            time.sleep(wait_time)

    # Now merge ALL data (existing + new) into race_results_all.csv
    print(f"\n{'=' * 60}")
    print("MERGING ALL DATA")
    print("=" * 60)

    all_frames = []
    for y in range(2014, 2026):
        fpath = RAW_DATA / "races" / f"race_results_{y}.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            all_frames.append(df)
            print(f"  {y}: {len(df)} entries, {df['round'].nunique()} races")

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        if "event_date" in merged.columns:
            merged["event_date"] = pd.to_datetime(merged["event_date"], format='mixed')
        merged = merged.sort_values(["year", "round"]).reset_index(drop=True)

        merged.to_csv(RAW_DATA / "races" / "race_results_all.csv", index=False)
        merged.to_pickle(RAW_DATA / "races" / "race_results_all.pkl")

        print(f"\n{'=' * 60}")
        print(f"FINAL: {len(merged)} entries, {merged['year'].nunique()} seasons, {merged['round'].nunique()} races")
        print(f"  Drivers: {merged['driver_abbreviation'].nunique()}")
        print(f"  Constructors: {merged['constructor'].nunique()}")
        print(f"  Circuits: {merged['circuit'].nunique()}")
        print(f"  Years: {sorted(merged['year'].unique())}")
        print(f"  Races per year:")
        for y in sorted(merged['year'].unique()):
            n_races = merged[merged['year'] == y]['round'].nunique()
            n_drivers = len(merged[merged['year'] == y])
            print(f"    {y}: {n_races} races, {n_drivers} entries")
        print("=" * 60)


if __name__ == "__main__":
    main()

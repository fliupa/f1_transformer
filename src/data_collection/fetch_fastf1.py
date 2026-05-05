"""Fetch F1 data from FastF1 API for all seasons 2014-2025."""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
F1_CACHE = PROJECT_ROOT / "data" / "f1_cache"

if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")
    F1_CACHE = Path("/content/drive/MyDrive/f1_transformer/data/f1_cache")

import fastf1
fastf1.Cache.enable_cache(str(F1_CACHE))


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

        # Get race session (try without weather first for older sessions)
        race = None
        try:
            race = fastf1.get_session(year, event_name, "R")
            race.load(telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"    Skipping race {year} {event_name}: {e}")
            continue

        if race is None or not hasattr(race, "results") or race.results is None:
            continue

        # Get qualifying session
        quali_data = {}
        try:
            quali = fastf1.get_session(year, event_name, "Q")
            quali.load(telemetry=False, weather=False, messages=False)
            quali_results = quali.results
            quali_data = {}
            for _, drv in quali_results.iterrows():
                key = drv.get("Abbreviation", drv.get("DriverNumber"))
                quali_data[key] = {
                    "q_position": drv.get("Position", np.nan),
                    "q1_time": _safe_td_seconds(drv.get("Q1")),
                    "q2_time": _safe_td_seconds(drv.get("Q2")),
                    "q3_time": _safe_td_seconds(drv.get("Q3")),
                }
        except Exception:
            quali_data = {}

        # Get sprint results if available
        sprint_data = {}
        try:
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

        # Process race results
        # Get circuit name safely
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
                # Rain flag
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

        print(f"  [{year}] Round {int(round_num)}: {event_name} - {len(race.results)} drivers")

    return results_list


def _safe_td_seconds(td):
    """Safely convert timedelta to seconds."""
    if td is None or pd.isna(td):
        return np.nan
    try:
        return td.total_seconds()
    except Exception:
        return np.nan


def main():
    """Fetch all F1 data from 2014 to 2025."""
    RAW_DATA.mkdir(parents=True, exist_ok=True)

    all_data = []
    years = list(range(2014, 2026))

    print("=" * 60)
    print("FETCHING F1 DATA 2014-2025 FROM FastF1")
    print("=" * 60)

    for year in years:
        print(f"\nSeason {year}...")
        season = fetch_season_results(year)
        all_data.extend(season)
        print(f"  Total: {len(season)} entries")

        # Save incremental
        year_df = pd.DataFrame([r for r in all_data if r["year"] == year])
        if not year_df.empty:
            year_df.to_csv(RAW_DATA / "races" / f"race_results_{year}.csv", index=False)

    # Save complete dataset
    df = pd.DataFrame(all_data)
    df.to_csv(RAW_DATA / "races" / "race_results_all.csv", index=False)
    df.to_pickle(RAW_DATA / "races" / "race_results_all.pkl")

    print("\n" + "=" * 60)
    print(f"DONE: {len(df)} entries across {df['year'].nunique()} seasons")
    print(f"  Races: {df.groupby('year')['round'].max().sum():.0f}")
    print(f"  Drivers: {df['driver_abbreviation'].nunique()}")
    print(f"  Constructors: {df['constructor'].nunique()}")
    print(f"  Circuits: {df['circuit'].nunique()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Fetch historical weather data from Open-Meteo API for all F1 circuits.

Open-Meteo is free and requires no API key.
Uses circuit GPS coordinates to query weather for each race weekend.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import os

try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    HAS_OPENMETEO = True
except ImportError:
    HAS_OPENMETEO = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")

# Circuit GPS coordinates (latitude, longitude)
CIRCUIT_COORDS = {
    "Albert Park": (-37.8497, 144.968),
    "Melbourne Grand Prix Circuit": (-37.8497, 144.968),
    "Albert Park Circuit": (-37.8497, 144.968),
    "Albert Park Grand Prix Circuit": (-37.8497, 144.968),
    "Sakhir": (26.0325, 50.5106),
    "Bahrain International Circuit": (26.0325, 50.5106),
    "Bahrain Outer Circuit": (26.0325, 50.5106),
    "Jeddah Corniche Circuit": (21.6319, 39.1044),
    "Autodromo Enzo e Dino Ferrari": (44.3412, 11.7129),
    "Imola Circuit": (44.3412, 11.7129),
    "Autodromo Nazionale Monza": (45.6208, 9.2812),
    "Monza Circuit": (45.6208, 9.2812),
    "Autodromo di Monza": (45.6208, 9.2812),
    "Circuit de Monaco": (43.7347, 7.4214),
    "Monaco Circuit": (43.7347, 7.4214),
    "Circuit de Barcelona-Catalunya": (41.5700, 2.2611),
    "Barcelona-Catalunya Circuit": (41.5700, 2.2611),
    "Circuit Gilles Villeneuve": (45.5000, -73.5228),
    "Red Bull Ring": (47.2197, 14.7647),
    "Silverstone Circuit": (52.0786, -1.0169),
    "Hungaroring": (47.5833, 19.2486),
    "Circuit de Spa-Francorchamps": (50.4372, 5.9714),
    "Spa-Francorchamps": (50.4372, 5.9714),
    "Circuit Park Zandvoort": (52.3889, 4.5409),
    "Zandvoort Circuit": (52.3889, 4.5409),
    "Circuit Zandvoort": (52.3889, 4.5409),
    "Autodromo Nazionale di Monza": (45.6208, 9.2812),
    "Monza": (45.6208, 9.2812),
    "Suzuka International Racing Course": (34.8431, 136.5408),
    "Suzuka Circuit": (34.8431, 136.5408),
    "Circuit of the Americas": (30.1328, -97.6411),
    "Autódromo Hermanos Rodríguez": (19.4050, -99.0936),
    "Mexico City Circuit": (19.4050, -99.0936),
    "Autódromo José Carlos Pace": (-23.7036, -46.6997),
    "Interlagos Circuit": (-23.7036, -46.6997),
    "Yas Marina Circuit": (24.4672, 54.6031),
    "Shanghai International Circuit": (31.3389, 121.2195),
    "Sochi Autodrom": (43.4087, 39.9709),
    "Baku City Circuit": (40.3725, 49.8533),
    "Marina Bay Street Circuit": (1.2918, 103.8642),
    "Sepang International Circuit": (2.7602, 101.7382),
    "Hockenheimring": (49.3300, 8.5696),
    "Nürburgring": (50.3356, 6.9475),
    "Istanbul Park": (40.9513, 29.4036),
    "Portimão Circuit": (37.2288, -8.6281),
    "Algarve International Circuit": (37.2288, -8.6281),
    "Mugello Circuit": (43.9976, 11.3719),
    "Autodromo Internazionale del Mugello": (43.9976, 11.3719),
    "Circuit Paul Ricard": (43.2507, 5.7917),
    "Paul Ricard Circuit": (43.2507, 5.7917),
    "Losail International Circuit": (25.4869, 51.4542),
    "Miami International Autodrome": (25.9581, -80.2389),
    "Las Vegas Strip Circuit": (36.1146, -115.1728),
    "Jerez Circuit": (36.7111, -6.0342),
    "Buriram International Circuit": (14.9622, 103.0911),
    "Chang International Circuit": (14.9622, 103.0911),
    "Korean International Circuit": (34.7366, 126.4139),
    "Valencia Street Circuit": (39.4589, -0.3317),
    "Buddh International Circuit": (28.3500, 77.5333),
    "Madrid Street Circuit": (40.4168, -3.7038),
    "Autódromo de Madrid": (40.4168, -3.7038),
    "Kuwait Motor Town": (29.3537, 47.6600),
    "Circuit Zolder": (50.9963, 5.2580),
}


def fetch_weather_for_race(circuit_name: str, event_date, client=None) -> dict:
    """Fetch weather for a race weekend at the given circuit."""
    coords = CIRCUIT_COORDS.get(circuit_name)
    if coords is None:
        # Try partial matching
        for name, c in CIRCUIT_COORDS.items():
            if circuit_name and name.lower() in circuit_name.lower():
                coords = c
                break
        if coords is None:
            return _empty_weather()

    lat, lon = coords
    if isinstance(event_date, str):
        event_date = pd.Timestamp(event_date)

    start_date = event_date - timedelta(days=3)  # Friday
    end_date = event_date  # Sunday race day

    # If Open-Meteo is available, fetch from API
    if HAS_OPENMETEO and client is not None:
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": [
                    "temperature_2m", "relative_humidity_2m", "precipitation",
                    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
                    "weather_code"
                ],
                "timezone": "auto",
            }
            responses = client.weather_api(
                "https://api.open-meteo.com/v1/forecast", params=params
            )
            response = responses[0]

            hourly = response.Hourly()
            if hourly is None:
                return _empty_weather()

            n_hours = len(hourly.Variables(0).ValuesAsNumpy())

            def _mean(var_idx):
                vals = hourly.Variables(var_idx).ValuesAsNumpy()
                return float(np.nanmean(vals)) if len(vals) > 0 else np.nan

            def _sum(var_idx):
                vals = hourly.Variables(var_idx).ValuesAsNumpy()
                return float(np.nansum(vals)) if len(vals) > 0 else 0.0

            temp_vals = hourly.Variables(0).ValuesAsNumpy()
            temp_mean = _mean(0)
            temp_max = float(np.nanmax(temp_vals)) if len(temp_vals) > 0 else np.nan
            temp_min = float(np.nanmin(temp_vals)) if len(temp_vals) > 0 else np.nan
            humidity = _mean(1)
            precip = _sum(2)
            wind_speed = _mean(3)
            wind_dir = _mean(4)
            pressure = _mean(5)

            weather_codes = hourly.Variables(6).ValuesAsNumpy()
            rain_codes = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 91, 92, 95, 96, 99}
            rain_flag = 1 if any(wc in rain_codes for wc in weather_codes) else 0

            return {
                "openmeteo_temp_mean": round(temp_mean, 1) if not np.isnan(temp_mean) else np.nan,
                "openmeteo_temp_max": round(temp_max, 1) if not np.isnan(temp_max) else np.nan,
                "openmeteo_temp_min": round(temp_min, 1) if not np.isnan(temp_min) else np.nan,
                "openmeteo_humidity": round(humidity, 1) if not np.isnan(humidity) else np.nan,
                "openmeteo_precip_mm": round(precip, 1),
                "openmeteo_wind_speed": round(wind_speed, 1) if not np.isnan(wind_speed) else np.nan,
                "openmeteo_wind_dir": round(wind_dir, 1) if not np.isnan(wind_dir) else np.nan,
                "openmeteo_pressure": round(pressure, 1) if not np.isnan(pressure) else np.nan,
                "openmeteo_rain_flag": rain_flag,
            }
        except Exception as e:
            print(f"    Open-Meteo error for {circuit_name}: {e}")

    return _empty_weather()


def _empty_weather():
    return {
        "openmeteo_temp_mean": np.nan,
        "openmeteo_temp_max": np.nan,
        "openmeteo_temp_min": np.nan,
        "openmeteo_humidity": np.nan,
        "openmeteo_precip_mm": np.nan,
        "openmeteo_wind_speed": np.nan,
        "openmeteo_wind_dir": np.nan,
        "openmeteo_pressure": np.nan,
        "openmeteo_rain_flag": 0,
    }


def main():
    """Fetch weather for all race events in the dataset."""
    RAW_DATA.mkdir(parents=True, exist_ok=True)
    (RAW_DATA / "weather").mkdir(exist_ok=True)

    # Try loading the race results
    race_file = RAW_DATA / "races" / "race_results_all.csv"
    if not race_file.exists():
        race_file = RAW_DATA / "races" / "race_results_all.pkl"
        if not race_file.exists():
            print("No race results found. Run fetch_fastf1.py first.")
            print("Generating weather for all circuits for all seasons anyway...")
            # Create a manual list of seasons and approximate dates
            return _fetch_weather_standalone()

    if str(race_file).endswith(".pkl"):
        df = pd.read_pickle(race_file)
    else:
        df = pd.read_csv(race_file)
        df["event_date"] = pd.to_datetime(df["event_date"])

    # Get unique race events
    events = df.groupby(["year", "circuit", "event_date"]).first().reset_index()
    events = events[["year", "circuit", "event_date"]].dropna(subset=["circuit", "event_date"])
    events = events.drop_duplicates(subset=["year", "circuit"])

    # Setup Open-Meteo client
    client = None
    if HAS_OPENMETEO:
        cache_session = requests_cache.CachedSession(
            str(RAW_DATA / "weather" / ".cache"), expire_after=3600
        )
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=retry_session)
        print("Open-Meteo client ready (with caching)")
    else:
        print("Open-Meteo not available. Install: pip install openmeteo-requests requests-cache retry-requests")

    print("=" * 60)
    print(f"FETCHING WEATHER FOR {len(events)} RACE EVENTS")
    print("=" * 60)

    weather_records = []
    for _, event in events.iterrows():
        circuit = event["circuit"]
        year = event["year"]
        date = pd.Timestamp(event["event_date"])
        print(f"  {year} - {circuit} ({date.date()})")

        weather = fetch_weather_for_race(circuit, date, client)
        weather["year"] = int(year)
        weather["circuit"] = circuit
        weather["event_date"] = date
        weather_records.append(weather)
        time.sleep(0.1)  # Be nice to the API

    weather_df = pd.DataFrame(weather_records)
    weather_df.to_csv(RAW_DATA / "weather" / "weather_all.csv", index=False)

    print(f"\nDone: {len(weather_df)} weather records")
    print(f"  Races with rain: {weather_df['openmeteo_rain_flag'].sum()}")
    print(f"  Temp range: {weather_df['openmeteo_temp_mean'].min():.0f}C - {weather_df['openmeteo_temp_mean'].max():.0f}C")


def _fetch_weather_standalone():
    """Fetch weather for all circuits for every year 2014-2025 without race data."""
    cache_session = requests_cache.CachedSession(
        str(RAW_DATA / "weather" / ".cache"), expire_after=3600 * 24 * 30
    )
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session) if HAS_OPENMETEO else None

    weather_records = []
    for name, (lat, lon) in sorted(CIRCUIT_COORDS.items()):
        print(f"  {name} ({lat}, {lon})")
        for year in range(2014, 2026):
            # Approximate race date: mid-season
            for month in [3, 5, 7, 9, 11]:
                date = pd.Timestamp(f"{year}-{month:02d}-15")
                weather = fetch_weather_for_race(name, date, client)
                weather["year"] = year
                weather["circuit"] = name
                weather["event_date"] = date
                weather_records.append(weather)
                time.sleep(0.05)

    weather_df = pd.DataFrame(weather_records)
    weather_df.to_csv(RAW_DATA / "weather" / "weather_all.csv", index=False)


if __name__ == "__main__":
    main()

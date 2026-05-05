"""Generate comprehensive mock F1 data for pipeline testing.

Creates realistic race results for 2014-2025 with all required features.
Uses real patterns: dominant teams/drivers, realistic points, DNF rates.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")

# Realistic F1 driver lineups per era
DRIVER_LINEUPS = {
    2014: ["HAM", "ROS", "VET", "RIC", "ALO", "RAI", "BUT", "MAS", "BOT", "MAG", "PER", "HUL", "VER", "GRO", "MAL", "GUT", "BIA", "KOB", "ERI", "CHI"],
    2015: ["HAM", "ROS", "VET", "RIC", "ALO", "RAI", "BUT", "MAS", "BOT", "MAG", "PER", "HUL", "VER", "GRO", "MAL", "NAS", "SAI", "KVY", "ERI", "STE"],
    2016: ["HAM", "ROS", "VET", "RIC", "ALO", "RAI", "BUT", "MAS", "BOT", "MAG", "PER", "HUL", "VER", "GRO", "PAL", "NAS", "SAI", "KVY", "ERI", "WEH"],
    2017: ["HAM", "BOT", "VET", "RIC", "ALO", "RAI", "VAN", "MAS", "VER", "MAG", "PER", "HUL", "SAI", "GRO", "PAL", "OCO", "STR", "KVY", "ERI", "WEH"],
    2018: ["HAM", "BOT", "VET", "RIC", "ALO", "RAI", "VAN", "LEC", "VER", "MAG", "PER", "HUL", "SAI", "GRO", "GAS", "OCO", "STR", "KVY", "ERI", "SIR"],
    2019: ["HAM", "BOT", "VET", "LEC", "RIC", "HUL", "ALB", "KVY", "VER", "GAS", "PER", "STR", "SAI", "NOR", "RAI", "GIO", "MAG", "GRO", "RUS", "KUB"],
    2020: ["HAM", "BOT", "VET", "LEC", "RIC", "OCO", "ALB", "VER", "PER", "STR", "SAI", "NOR", "RAI", "GIO", "MAG", "GRO", "GAS", "KVY", "RUS", "LAT"],
    2021: ["HAM", "BOT", "VER", "PER", "LEC", "SAI", "RIC", "NOR", "ALO", "OCO", "GAS", "TSU", "VET", "STR", "RAI", "GIO", "RUS", "LAT", "MSC", "MAZ"],
    2022: ["HAM", "RUS", "VER", "PER", "LEC", "SAI", "RIC", "NOR", "ALO", "OCO", "GAS", "TSU", "VET", "STR", "BOT", "ZHO", "MAG", "MSC", "ALB", "LAT"],
    2023: ["HAM", "RUS", "VER", "PER", "LEC", "SAI", "NOR", "PIA", "ALO", "STR", "GAS", "OCO", "BOT", "ZHO", "MAG", "HUL", "TSU", "DEV", "ALB", "SAR"],
    2024: ["HAM", "RUS", "VER", "PER", "LEC", "SAI", "NOR", "PIA", "ALO", "STR", "GAS", "OCO", "BOT", "ZHO", "MAG", "HUL", "TSU", "RIC", "ALB", "SAR"],
    2025: ["HAM", "LEC", "VER", "NOR", "PIA", "RUS", "SAI", "ALB", "ALO", "STR", "GAS", "OCO", "BOT", "ZHO", "MAG", "HUL", "TSU", "RIC", "BEA", "DOO"],
}

CONSTRUCTOR_MAP = {
    "HAM": {2014: "Mercedes", 2015: "Mercedes", 2016: "Mercedes", 2017: "Mercedes", 2018: "Mercedes", 2019: "Mercedes", 2020: "Mercedes", 2021: "Mercedes", 2022: "Mercedes", 2023: "Mercedes", 2024: "Mercedes", 2025: "Ferrari"},
    "ROS": {2014: "Mercedes", 2015: "Mercedes", 2016: "Mercedes"},
    "BOT": {2017: "Mercedes", 2018: "Mercedes", 2019: "Mercedes", 2020: "Mercedes", 2021: "Mercedes", 2022: "Alfa Romeo", 2023: "Alfa Romeo", 2024: "Kick Sauber", 2025: "Mercedes"},
    "VET": {2014: "Red Bull", 2015: "Ferrari", 2016: "Ferrari", 2017: "Ferrari", 2018: "Ferrari", 2019: "Ferrari", 2020: "Ferrari", 2021: "Aston Martin", 2022: "Aston Martin"},
    "RIC": {2014: "Red Bull", 2015: "Red Bull", 2016: "Red Bull", 2017: "Red Bull", 2018: "Red Bull", 2019: "Renault", 2020: "Renault", 2021: "McLaren", 2022: "McLaren", 2023: "AlphaTauri", 2024: "RB", 2025: "RB"},
    "ALO": {2014: "Ferrari", 2015: "McLaren", 2016: "McLaren", 2017: "McLaren", 2018: "McLaren", 2021: "Alpine", 2022: "Alpine", 2023: "Aston Martin", 2024: "Aston Martin", 2025: "Aston Martin"},
    "RAI": {2014: "Ferrari", 2015: "Ferrari", 2016: "Ferrari", 2017: "Ferrari", 2018: "Ferrari", 2019: "Alfa Romeo", 2020: "Alfa Romeo", 2021: "Alfa Romeo"},
    "BUT": {2014: "McLaren", 2015: "McLaren", 2016: "McLaren"},
    "MAS": {2014: "Williams", 2015: "Williams", 2016: "Williams", 2017: "Williams"},
    "VER": {2015: "Toro Rosso", 2016: "Red Bull", 2017: "Red Bull", 2018: "Red Bull", 2019: "Red Bull", 2020: "Red Bull", 2021: "Red Bull", 2022: "Red Bull", 2023: "Red Bull", 2024: "Red Bull", 2025: "Red Bull"},
    "LEC": {2018: "Sauber", 2019: "Ferrari", 2020: "Ferrari", 2021: "Ferrari", 2022: "Ferrari", 2023: "Ferrari", 2024: "Ferrari", 2025: "Ferrari"},
    "NOR": {2019: "McLaren", 2020: "McLaren", 2021: "McLaren", 2022: "McLaren", 2023: "McLaren", 2024: "McLaren", 2025: "McLaren"},
    "RUS": {2019: "Williams", 2020: "Williams", 2021: "Williams", 2022: "Mercedes", 2023: "Mercedes", 2024: "Mercedes", 2025: "Mercedes"},
    "PER": {2014: "Force India", 2015: "Force India", 2016: "Force India", 2017: "Force India", 2018: "Force India", 2019: "Racing Point", 2020: "Racing Point", 2021: "Red Bull", 2022: "Red Bull", 2023: "Red Bull", 2024: "Red Bull"},
    "SAI": {2015: "Toro Rosso", 2016: "Toro Rosso", 2017: "Toro Rosso", 2018: "Renault", 2019: "McLaren", 2020: "McLaren", 2021: "Ferrari", 2022: "Ferrari", 2023: "Ferrari", 2024: "Ferrari", 2025: "Williams"},
    "GAS": {2017: "Toro Rosso", 2018: "Toro Rosso", 2019: "Red Bull", 2020: "AlphaTauri", 2021: "AlphaTauri", 2022: "AlphaTauri", 2023: "Alpine", 2024: "Alpine", 2025: "Alpine"},
    "STR": {2017: "Williams", 2018: "Williams", 2019: "Racing Point", 2020: "Racing Point", 2021: "Aston Martin", 2022: "Aston Martin", 2023: "Aston Martin", 2024: "Aston Martin", 2025: "Aston Martin"},
    "OCO": {2016: "Manor", 2017: "Force India", 2018: "Force India", 2020: "Renault", 2021: "Alpine", 2022: "Alpine", 2023: "Alpine", 2024: "Alpine", 2025: "Haas"},
    "MAG": {2014: "McLaren", 2015: "McLaren", 2016: "Renault", 2017: "Haas", 2018: "Haas", 2019: "Haas", 2020: "Haas", 2022: "Haas", 2023: "Haas", 2024: "Haas", 2025: "Haas"},
    "HUL": {2014: "Force India", 2015: "Force India", 2016: "Force India", 2017: "Renault", 2018: "Renault", 2019: "Renault", 2020: "Racing Point", 2022: "Aston Martin", 2023: "Haas", 2024: "Haas", 2025: "Kick Sauber"},
    "GRO": {2014: "Lotus", 2015: "Lotus", 2016: "Haas", 2017: "Haas", 2018: "Haas", 2019: "Haas", 2020: "Haas"},
    "ALB": {2019: "Toro Rosso", 2020: "Red Bull", 2021: "Williams", 2023: "Williams", 2024: "Williams", 2025: "Williams"},
    "TSU": {2021: "AlphaTauri", 2022: "AlphaTauri", 2023: "AlphaTauri", 2024: "RB", 2025: "RB"},
    "PIA": {2023: "McLaren", 2024: "McLaren", 2025: "McLaren"},
    "ZHO": {2022: "Alfa Romeo", 2023: "Alfa Romeo", 2024: "Kick Sauber", 2025: "Ferrari"},
}

# Circuit list with realistic calendar order
CIRCUITS_PER_YEAR = {
    2014: ["Albert Park", "Sepang", "Bahrain", "Shanghai", "Barcelona", "Monaco", "Montreal", "Red Bull Ring", "Silverstone", "Hockenheim", "Hungaroring", "Spa", "Monza", "Singapore", "Suzuka", "Sochi", "Austin", "Interlagos", "Yas Marina"],
    2015: ["Albert Park", "Sepang", "Shanghai", "Bahrain", "Barcelona", "Monaco", "Montreal", "Red Bull Ring", "Silverstone", "Hungaroring", "Spa", "Monza", "Singapore", "Suzuka", "Sochi", "Austin", "Mexico City", "Interlagos", "Yas Marina"],
    2016: ["Albert Park", "Bahrain", "Shanghai", "Sochi", "Barcelona", "Monaco", "Montreal", "Baku", "Red Bull Ring", "Silverstone", "Hungaroring", "Hockenheim", "Spa", "Monza", "Singapore", "Sepang", "Suzuka", "Austin", "Mexico City", "Interlagos", "Yas Marina"],
    2017: ["Albert Park", "Shanghai", "Bahrain", "Sochi", "Barcelona", "Monaco", "Montreal", "Baku", "Red Bull Ring", "Silverstone", "Hungaroring", "Spa", "Monza", "Singapore", "Sepang", "Suzuka", "Austin", "Mexico City", "Interlagos", "Yas Marina"],
    2018: ["Albert Park", "Bahrain", "Shanghai", "Baku", "Barcelona", "Monaco", "Montreal", "Paul Ricard", "Red Bull Ring", "Silverstone", "Hockenheim", "Hungaroring", "Spa", "Monza", "Singapore", "Sochi", "Suzuka", "Austin", "Mexico City", "Interlagos", "Yas Marina"],
    2019: ["Albert Park", "Bahrain", "Shanghai", "Baku", "Barcelona", "Monaco", "Montreal", "Paul Ricard", "Red Bull Ring", "Silverstone", "Hockenheim", "Hungaroring", "Spa", "Monza", "Singapore", "Sochi", "Suzuka", "Mexico City", "Austin", "Interlagos", "Yas Marina"],
    2020: ["Red Bull Ring", "Red Bull Ring", "Hungaroring", "Silverstone", "Silverstone", "Barcelona", "Spa", "Monza", "Mugello", "Sochi", "Nurburgring", "Portimao", "Imola", "Istanbul", "Bahrain", "Bahrain", "Yas Marina"],
    2021: ["Bahrain", "Imola", "Portimao", "Barcelona", "Monaco", "Baku", "Paul Ricard", "Red Bull Ring", "Red Bull Ring", "Hungaroring", "Spa", "Zandvoort", "Monza", "Sochi", "Istanbul", "Austin", "Mexico City", "Interlagos", "Losail", "Jeddah", "Yas Marina"],
    2022: ["Bahrain", "Jeddah", "Albert Park", "Imola", "Miami", "Barcelona", "Monaco", "Baku", "Montreal", "Silverstone", "Red Bull Ring", "Paul Ricard", "Hungaroring", "Spa", "Zandvoort", "Monza", "Singapore", "Suzuka", "Austin", "Mexico City", "Interlagos", "Yas Marina"],
    2023: ["Bahrain", "Jeddah", "Albert Park", "Baku", "Miami", "Imola", "Monaco", "Barcelona", "Montreal", "Red Bull Ring", "Silverstone", "Hungaroring", "Spa", "Zandvoort", "Monza", "Singapore", "Suzuka", "Losail", "Austin", "Mexico City", "Interlagos", "Las Vegas", "Yas Marina"],
    2024: ["Bahrain", "Jeddah", "Albert Park", "Suzuka", "Shanghai", "Miami", "Imola", "Monaco", "Montreal", "Barcelona", "Red Bull Ring", "Silverstone", "Hungaroring", "Spa", "Zandvoort", "Monza", "Baku", "Singapore", "Austin", "Mexico City", "Interlagos", "Las Vegas", "Losail", "Yas Marina"],
    2025: ["Albert Park", "Shanghai", "Suzuka", "Bahrain", "Jeddah", "Miami", "Imola", "Monaco", "Barcelona", "Montreal", "Red Bull Ring", "Silverstone", "Spa", "Hungaroring", "Zandvoort", "Monza", "Baku", "Singapore", "Austin", "Mexico City", "Interlagos", "Las Vegas", "Losail", "Yas Marina"],
}

# Car performance tiers per season (dict: constructor -> tier 1-5, lower = better)
TIERS = {
    2014: {"Mercedes": 1, "Red Bull": 2, "Williams": 2, "Ferrari": 3, "McLaren": 3, "Force India": 3, "Toro Rosso": 4, "Lotus": 4, "Marussia": 5, "Caterham": 5, "Sauber": 5},
    2015: {"Mercedes": 1, "Ferrari": 2, "Williams": 3, "Red Bull": 3, "Force India": 4, "Lotus": 4, "Toro Rosso": 4, "McLaren": 5, "Marussia": 5, "Sauber": 5},
    2016: {"Mercedes": 1, "Red Bull": 2, "Ferrari": 3, "Force India": 3, "Williams": 3, "McLaren": 4, "Toro Rosso": 4, "Haas": 4, "Renault": 5, "Manor": 5, "Sauber": 5},
    2017: {"Mercedes": 1, "Ferrari": 1, "Red Bull": 2, "Force India": 3, "Williams": 4, "Renault": 4, "Toro Rosso": 4, "Haas": 4, "McLaren": 5, "Sauber": 5},
    2018: {"Mercedes": 1, "Ferrari": 1, "Red Bull": 2, "Renault": 3, "Haas": 3, "Force India": 3, "McLaren": 4, "Toro Rosso": 4, "Sauber": 4, "Williams": 5},
    2019: {"Mercedes": 1, "Ferrari": 2, "Red Bull": 2, "McLaren": 3, "Renault": 3, "Toro Rosso": 4, "Racing Point": 4, "Alfa Romeo": 4, "Haas": 5, "Williams": 5},
    2020: {"Mercedes": 1, "Red Bull": 2, "Racing Point": 2, "McLaren": 3, "Renault": 3, "Ferrari": 4, "AlphaTauri": 4, "Alfa Romeo": 5, "Haas": 5, "Williams": 5},
    2021: {"Mercedes": 1, "Red Bull": 1, "McLaren": 2, "Ferrari": 3, "Alpine": 3, "AlphaTauri": 3, "Aston Martin": 4, "Williams": 4, "Alfa Romeo": 5, "Haas": 5},
    2022: {"Red Bull": 1, "Ferrari": 1, "Mercedes": 2, "Alpine": 3, "McLaren": 3, "Alfa Romeo": 4, "Aston Martin": 4, "Haas": 4, "AlphaTauri": 4, "Williams": 5},
    2023: {"Red Bull": 1, "Ferrari": 2, "Mercedes": 2, "Aston Martin": 2, "McLaren": 3, "Alpine": 4, "Williams": 4, "AlphaTauri": 4, "Alfa Romeo": 5, "Haas": 5},
    2024: {"Red Bull": 1, "McLaren": 1, "Ferrari": 1, "Mercedes": 2, "Aston Martin": 3, "RB": 4, "Haas": 4, "Alpine": 4, "Williams": 5, "Kick Sauber": 5},
    2025: {"McLaren": 1, "Ferrari": 1, "Red Bull": 1, "Mercedes": 2, "Aston Martin": 3, "Alpine": 3, "RB": 4, "Haas": 4, "Williams": 4, "Kick Sauber": 5},
}

# Dominant drivers per season and their win probability
DOMINANT = {
    2014: {"HAM": 0.40, "ROS": 0.30}, 2015: {"HAM": 0.45, "ROS": 0.25},
    2016: {"HAM": 0.35, "ROS": 0.30}, 2017: {"HAM": 0.35, "VET": 0.25},
    2018: {"HAM": 0.40, "VET": 0.20}, 2019: {"HAM": 0.40, "BOT": 0.15},
    2020: {"HAM": 0.45, "BOT": 0.10, "VER": 0.10},
    2021: {"VER": 0.35, "HAM": 0.30},
    2022: {"VER": 0.50, "LEC": 0.15},
    2023: {"VER": 0.65, "PER": 0.08},
    2024: {"VER": 0.30, "NOR": 0.20, "LEC": 0.10},
    2025: {"NOR": 0.20, "LEC": 0.18, "VER": 0.15, "PIA": 0.12, "HAM": 0.10},
}


def generate_season(year):
    """Generate realistic race results for one season."""
    drivers = DRIVER_LINEUPS[year]
    circuits = CIRCUITS_PER_YEAR[year]
    tiers = TIERS[year]
    dominant = DOMINANT[year]
    np.random.seed(year * 100 + 42)

    results = []
    for round_num, circuit in enumerate(circuits, 1):
        event_date = pd.Timestamp(f"{year}-{3 + (round_num//4):02d}-{(round_num%4)*7 + 1:02d}")
        event_name = f"{circuit} Grand Prix"

        # Weather
        temp = np.random.normal(28, 8)
        humidity = np.random.normal(55, 20)
        humidity = max(10, min(100, humidity))
        precip = np.random.exponential(0.5) if np.random.random() < 0.2 else 0
        wind = np.random.normal(15, 8)
        rain = 1 if precip > 1.0 else 0
        pressure = np.random.normal(1013, 10)

        # Driver positions generation
        driver_data = []
        for driver in drivers:
            const = CONSTRUCTOR_MAP.get(driver, {}).get(year, "Unknown")
            tier = tiers.get(const, 5)

            # Base pace: tier affects position
            tier_offset = (tier - 1) * 4
            driver_skill = np.random.normal(0, 2)

            # Dominant driver boost
            dom_bonus = 0
            if driver in dominant:
                if np.random.random() < dominant[driver]:
                    dom_bonus = -np.random.randint(0, 3)

            grid_pos = max(1, min(20, int(np.random.normal(tier_offset + 2 + driver_skill + dom_bonus, 4))))

            # Qualifying position
            q_pos = max(1, min(20, int(np.random.normal(grid_pos, 2))))
            q1_time = 80 + np.random.normal(tier_offset, 2) + np.random.normal(0, 0.5)
            q2_time = q1_time - np.random.normal(0.5, 0.2)
            q3_time = q2_time - np.random.normal(0.3, 0.1) if np.random.random() < 0.5 else None

            # Finish position
            finish_base = grid_pos + np.random.normal(-1 * (5 - tier), 3)
            finish_pos = max(1, min(20, int(finish_base)))
            dnf = 1 if np.random.random() < 0.08 else 0

            # Points
            points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
            points = points_system[finish_pos - 1] if finish_pos <= 10 and dnf == 0 else 0

            driver_data.append({
                "grid_position": grid_pos, "q_position": q_pos,
                "q1_time": q1_time, "q2_time": q2_time, "q3_time": q3_time,
                "finish_position": finish_pos, "points": points, "dnf": dnf,
                "constructor": const,
            })

        # Sort by finish position for realistic results
        driver_data.sort(key=lambda x: (x["dnf"], x["finish_position"]))

        for i, dd in enumerate(driver_data):
            driver = drivers[i]
            dd2 = driver_data[i]
            status = "Finished" if dd2["dnf"] == 0 else "Engine"

            results.append({
                "year": year, "round": round_num, "event_name": event_name,
                "country": "Unknown", "event_date": event_date, "circuit": circuit,
                "driver_abbreviation": driver, "driver_number": i + 1,
                "constructor": dd2["constructor"],
                "grid_position": dd2["grid_position"], "finish_position": dd2["finish_position"],
                "points": dd2["points"], "dnf": dd2["dnf"], "status": status,
                "q_position": dd2["q_position"],
                "q1_time": dd2["q1_time"], "q2_time": dd2["q2_time"], "q3_time": dd2["q3_time"],
                "air_temp": temp, "track_temp": temp + np.random.normal(10, 5),
                "humidity": humidity, "rainfall": precip, "wind_speed": wind,
                "rain_flag": rain, "pressure": pressure,
            })

    df = pd.DataFrame(results)
    # Ensure unique finish positions per race
    for (yr, rnd), group in df.groupby(["year", "round"]):
        indices = group.index
        positions = list(range(1, len(group) + 1))
        survivors = positions[:len(indices)]
        survivors.sort()
        df.loc[indices, "finish_position"] = survivors
        # Reassign points
        for idx in indices:
            pos = df.loc[idx, "finish_position"]
            if pos <= 10 and df.loc[idx, "dnf"] == 0:
                df.loc[idx, "points"] = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1][pos - 1]
            else:
                df.loc[idx, "points"] = 0

    return df


def main():
    """Generate mock F1 data for 2014-2025."""
    RAW_DATA.mkdir(parents=True, exist_ok=True)
    (RAW_DATA / "races").mkdir(exist_ok=True)

    # Load real data if available
    real_df = None
    real_path = RAW_DATA / "races" / "race_results_all.csv"
    if real_path.exists():
        real_df = pd.read_csv(real_path)
        if "event_date" in real_df.columns:
            real_df["event_date"] = pd.to_datetime(real_df["event_date"])
        real_years = real_df["year"].unique()
        print(f"Found real data: {len(real_df)} entries for {real_years}")

    all_data = []
    years_to_generate = [y for y in range(2014, 2026) if real_df is None or y not in real_df["year"].values]

    if not years_to_generate:
        print("All years already have data. Using existing data.")
        df = real_df
    else:
        print(f"Generating mock data for {years_to_generate}...")
        for year in years_to_generate:
            df = generate_season(year)
            all_data.append(df)
            print(f"  {year}: {len(df)} entries, {df['round'].nunique()} races")

        mock_df = pd.DataFrame()
        if all_data:
            mock_df = pd.concat(all_data, ignore_index=True)

        # Merge with real data if available
        if real_df is not None:
            # Ensure same columns
            common_cols = set(real_df.columns) & set(mock_df.columns)
            real_subset = real_df[list(common_cols)]
            mock_subset = mock_df[list(common_cols)]
            df = pd.concat([real_subset, mock_subset], ignore_index=True)
        else:
            df = mock_df

    df["event_date"] = pd.to_datetime(df["event_date"])

    # Save
    df.to_csv(RAW_DATA / "races" / "race_results_all.csv", index=False)
    df.to_pickle(RAW_DATA / "races" / "race_results_all.pkl")

    print(f"\nDone! {len(df)} entries, {df['year'].nunique()} seasons, {df['round'].nunique()} races")
    print(f"  Drivers: {df['driver_abbreviation'].nunique()}")
    print(f"  Constructors: {df['constructor'].nunique()}")
    print(f"  Circuits: {df['circuit'].nunique()}")


if __name__ == "__main__":
    main()

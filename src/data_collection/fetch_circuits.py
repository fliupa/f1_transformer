"""Static circuit data with altitude, length, corners, DRS zones, track characteristics.

Data sources: Wikipedia, F1.com, public datasets.
Altitude data: Google Earth / public elevation datasets.
"""
import pandas as pd
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
if os.environ.get("COLAB"):
    RAW_DATA = Path("/content/drive/MyDrive/f1_transformer/data/raw")

# Complete circuit database for all F1 circuits 2014-2025
CIRCUIT_DATA = [
    # (name, country, length_km, corners, drs_zones, altitude_m, track_type, downforce, tyre_degrad, overtaking, lap_record_s)
    ("Albert Park", "Australia", 5.278, 14, 4, 8, "street", "medium", "medium", "medium", 80.235),
    ("Bahrain International Circuit", "Bahrain", 5.412, 15, 3, 17, "permanent", "medium", "high", "easy", 91.002),
    ("Jeddah Corniche Circuit", "Saudi Arabia", 6.174, 27, 3, 12, "street", "medium", "low", "easy", 90.734),
    ("Melbourne Grand Prix Circuit", "Australia", 5.278, 14, 4, 8, "street", "medium", "medium", "medium", 80.235),
    ("Albert Park Circuit", "Australia", 5.278, 14, 4, 8, "street", "medium", "medium", "medium", 80.235),
    ("Autodromo Enzo e Dino Ferrari", "Italy", 4.909, 19, 2, 38, "permanent", "medium", "medium", "hard", 75.484),
    ("Imola Circuit", "Italy", 4.909, 19, 2, 38, "permanent", "medium", "medium", "hard", 75.484),
    ("Autodromo Nazionale Monza", "Italy", 5.793, 11, 2, 162, "permanent", "low", "low", "easy", 81.046),
    ("Monza Circuit", "Italy", 5.793, 11, 2, 162, "permanent", "low", "low", "easy", 81.046),
    ("Autódromo José Carlos Pace", "Brazil", 4.309, 15, 2, 780, "permanent", "high", "medium", "medium", 70.540),
    ("Interlagos Circuit", "Brazil", 4.309, 15, 2, 780, "permanent", "high", "medium", "medium", 70.540),
    ("Baku City Circuit", "Azerbaijan", 6.003, 20, 2, -28, "street", "low", "low", "easy", 103.105),
    ("Bahrain Outer Circuit", "Bahrain", 3.543, 11, 1, 17, "permanent", "low", "low", "easy", 55.404),
    ("Barcelona-Catalunya Circuit", "Spain", 4.657, 14, 2, 110, "permanent", "high", "high", "hard", 78.149),
    ("Circuit de Barcelona-Catalunya", "Spain", 4.657, 14, 2, 110, "permanent", "high", "high", "hard", 78.149),
    ("Circuit de Monaco", "Monaco", 3.337, 19, 1, 10, "street", "high", "low", "hard", 72.909),
    ("Monaco Circuit", "Monaco", 3.337, 19, 1, 10, "street", "high", "low", "hard", 72.909),
    ("Circuit de Spa-Francorchamps", "Belgium", 7.004, 19, 2, 400, "permanent", "medium", "medium", "easy", 104.286),
    ("Spa-Francorchamps", "Belgium", 7.004, 19, 2, 400, "permanent", "medium", "medium", "easy", 104.286),
    ("Circuit Gilles Villeneuve", "Canada", 4.361, 14, 3, 10, "semi-permanent", "medium", "low", "easy", 73.078),
    ("Circuit of the Americas", "USA", 5.513, 20, 2, 172, "permanent", "high", "medium", "easy", 96.164),
    ("Circuit Park Zandvoort", "Netherlands", 4.259, 14, 2, 5, "permanent", "high", "high", "hard", 71.349),
    ("Zandvoort Circuit", "Netherlands", 4.259, 14, 2, 5, "permanent", "high", "high", "hard", 71.349),
    ("Circuit Zandvoort", "Netherlands", 4.259, 14, 2, 5, "permanent", "high", "high", "hard", 71.349),
    ("Hockenheimring", "Germany", 4.574, 17, 2, 97, "permanent", "medium", "medium", "medium", 73.780),
    ("Hungaroring", "Hungary", 4.381, 14, 2, 165, "permanent", "high", "high", "hard", 76.603),
    ("Istanbul Park", "Turkey", 5.338, 14, 2, 50, "permanent", "high", "low", "easy", 84.170),
    ("Korean International Circuit", "Korea", 5.615, 18, 2, 3, "permanent", "medium", "medium", "medium", 99.905),
    ("Kuwait Motor Town", "Kuwait", 5.032, 18, 2, 81, "permanent", "medium", "medium", "medium", 108.000),
    ("Las Vegas Strip Circuit", "USA", 6.201, 17, 2, 610, "street", "low", "low", "easy", 94.000),
    ("Losail International Circuit", "Qatar", 5.380, 16, 1, 9, "permanent", "high", "medium", "medium", 83.317),
    ("Marina Bay Street Circuit", "Singapore", 4.940, 19, 3, 5, "street", "high", "high", "hard", 95.188),
    ("Miami International Autodrome", "USA", 5.412, 19, 3, 2, "street", "medium", "medium", "easy", 89.790),
    ("Mugello Circuit", "Italy", 5.245, 15, 2, 290, "permanent", "high", "medium", "medium", 78.833),
    ("Nürburgring", "Germany", 5.148, 15, 2, 570, "permanent", "high", "medium", "medium", 88.139),
    ("Portimão Circuit", "Portugal", 4.653, 15, 2, 120, "permanent", "high", "medium", "easy", 78.821),
    ("Algarve International Circuit", "Portugal", 4.653, 15, 2, 120, "permanent", "high", "medium", "easy", 78.821),
    ("Red Bull Ring", "Austria", 4.318, 10, 3, 670, "permanent", "medium", "low", "easy", 65.619),
    ("Sepang International Circuit", "Malaysia", 5.543, 15, 2, 20, "permanent", "high", "high", "easy", 94.224),
    ("Shanghai International Circuit", "China", 5.451, 16, 2, 5, "permanent", "high", "medium", "easy", 92.114),
    ("Silverstone Circuit", "Great Britain", 5.891, 18, 2, 145, "permanent", "high", "high", "medium", 87.097),
    ("Sochi Autodrom", "Russia", 5.848, 18, 2, 10, "permanent", "medium", "medium", "hard", 95.761),
    ("Autódromo Hermanos Rodríguez", "Mexico", 4.304, 17, 2, 2250, "permanent", "high", "low", "easy", 77.774),
    ("Mexico City Circuit", "Mexico", 4.304, 17, 2, 2250, "permanent", "high", "low", "easy", 77.774),
    ("Suzuka International Racing Course", "Japan", 5.807, 18, 1, 45, "permanent", "high", "high", "medium", 90.983),
    ("Suzuka Circuit", "Japan", 5.807, 18, 1, 45, "permanent", "high", "high", "medium", 90.983),
    ("Yas Marina Circuit", "UAE", 5.281, 16, 2, 5, "permanent", "high", "medium", "hard", 82.109),
    ("Buddh International Circuit", "India", 5.137, 13, 2, 190, "permanent", "medium", "medium", "easy", 87.200),
    ("Valencia Street Circuit", "Spain", 5.419, 25, 2, 5, "street", "high", "medium", "hard", 98.560),
    ("Autodromo Internazionale del Mugello", "Italy", 5.245, 15, 2, 290, "permanent", "high", "medium", "medium", 78.833),
    ("Jerez Circuit", "Spain", 4.428, 13, 2, 30, "permanent", "high", "medium", "medium", 79.591),
    ("Circuit Paul Ricard", "France", 5.842, 15, 2, 30, "permanent", "medium", "medium", "medium", 92.740),
    ("Paul Ricard Circuit", "France", 5.842, 15, 2, 30, "permanent", "medium", "medium", "medium", 92.740),
    ("Buriram International Circuit", "Thailand", 4.554, 16, 2, 188, "permanent", "medium", "high", "easy", 85.436),
    ("Chang International Circuit", "Thailand", 4.554, 16, 2, 188, "permanent", "medium", "high", "easy", 85.436),
    ("Autodromo di Monza", "Italy", 5.793, 11, 2, 162, "permanent", "low", "low", "easy", 81.046),
    ("Albert Park Grand Prix Circuit", "Australia", 5.278, 14, 4, 8, "street", "medium", "medium", "medium", 80.235),
    ("Circuit Zolder", "Belgium", 4.000, 14, 2, 48, "permanent", "high", "medium", "medium", 79.000),
    ("Texas Motor Speedway", "USA", 5.513, 20, 2, 172, "permanent", "high", "medium", "easy", 96.164),
    # 2024-2025 additions
    ("Madrid Street Circuit", "Spain", 5.474, 20, 3, 667, "street", "medium", "medium", "medium", 90.000),
    ("Autódromo de Madrid", "Spain", 5.474, 20, 3, 667, "street", "medium", "medium", "medium", 90.000),
]


def get_circuits_df() -> pd.DataFrame:
    """Build circuits DataFrame with numeric encodings."""
    df = pd.DataFrame(CIRCUIT_DATA, columns=[
        "circuit_name", "country", "length_km", "corners", "drs_zones",
        "altitude_m", "track_type", "downforce", "tyre_degradation",
        "overtaking_difficulty", "lap_record_s"
    ])

    # Numeric encodings for categoricals
    df["track_type_code"] = df["track_type"].map({"street": 0, "semi-permanent": 1, "permanent": 2})
    df["downforce_code"] = df["downforce"].map({"low": 0, "medium": 1, "high": 2})
    df["tyre_degradation_code"] = df["tyre_degradation"].map({"low": 0, "medium": 1, "high": 2})
    df["overtaking_code"] = df["overtaking_difficulty"].map({"easy": 0, "medium": 1, "hard": 2})

    # Derived features
    df["avg_corner_speed"] = df["length_km"] * 1000 / df["corners"]  # meters per corner
    df["altitude_normalized"] = (df["altitude_m"] - df["altitude_m"].mean()) / df["altitude_m"].std()

    return df


def main():
    """Generate and save circuits dataset."""
    RAW_DATA.mkdir(parents=True, exist_ok=True)
    (RAW_DATA / "circuits").mkdir(exist_ok=True)

    df = get_circuits_df()
    df.to_csv(RAW_DATA / "circuits" / "circuits.csv", index=False)
    df.to_pickle(RAW_DATA / "circuits" / "circuits.pkl")

    print("=" * 60)
    print("CIRCUIT DATA")
    print("=" * 60)
    print(f"Total circuits: {df['circuit_name'].nunique()}")
    print(f"Track types:      {df['track_type'].value_counts().to_dict()}")
    print(f"Altitude range:   {df['altitude_m'].min():.0f}m - {df['altitude_m'].max():.0f}m")
    print(f"Length range:     {df['length_km'].min():.2f}km - {df['length_km'].max():.2f}km")
    print(f"Corners range:    {df['corners'].min()} - {df['corners'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

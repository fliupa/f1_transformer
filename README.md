# F1 Winner Prediction with Transformer

Modelo Deep Learning que predice el ganador de cada carrera de Formula 1 usando un Transformer dual-stream con cross-attention.

## Arquitectura

**Dual-Stream Cross-Attention Transformer** (2.28M parametros):

1. **Race History Encoder**: Transformer encoder procesa las ultimas N=10 carreras
2. **Driver Candidate Encoder**: MLP codifica features de cada piloto para la carrera objetivo
3. **Cross-Attention**: Los pilotos (queries) atienden al historial de carreras (keys/values)
4. **Prediction Head**: Produce probabilidades de victoria sobre 20 pilotos

Features por piloto: driver embedding (16d), constructor embedding (8d), circuit embedding (8d), + 10 features numericas (grid position, form reciente, puntos, historial en circuito, etc.)

## Resultados

| Set | Accuracy |
|-----|----------|
| Validacion 2014-2024 | 54.5% |
| **Test 2025 (24 carreras)** | **41.7% (10/24)** |

Baselines: random 5%, champion leader ~20%, pole position ~38-42%

Predicciones 2025 vs Realidad:
- VER: 12 pred / 8 real
- NOR: 8 pred / 7 real
- PIA: 1 pred / 7 real (breakout subestimado)
- HAM: 3 pred / 0 real
- RUS: 0 pred / 2 real

## Setup Rapido

```bash
git clone https://github.com/USERNAME/f1_transformer.git
cd f1_transformer
pip install -r requirements.txt
```

## Uso

```bash
# 1. Descargar datos (FastF1 API + Open-Meteo clima)
python scripts/fetch_data.py

# 2. Construir features y secuencias
python scripts/build_dataset.py

# 3. Entrenar modelo (2014-2024 combinado)
python scripts/train.py

# 4. Predecir 2025
python scripts/predict.py
```

## Google Colab

Abrir los notebooks en orden (00 al 04). Cada notebook clona el repo y ejecuta las celdas secuencialmente.

## Estructura del Proyecto

```
f1_transformer/
├── src/
│   ├── data_collection/   # FastF1, clima, circuitos
│   ├── preprocessing/     # Features, encoders, secuencias
│   ├── model/             # Transformer + positional encoding
│   ├── training/          # Trainer, losses, metrics
│   ├── prediction/        # Prediccion 2025
│   └── utils/             # Config, helpers
├── scripts/               # Entry points
├── notebooks/             # Colab notebooks
├── config/                # YAML configs
├── data/                  # Raw + processed (git-ignored)
├── models/                # Checkpoints (git-ignored)
└── outputs/               # Predictions + plots (git-ignored)
```

## Dependencias

- torch >= 2.0
- fastf1 >= 3.4
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- pyyaml, tqdm

## Key Insight

Fine-tuning separado (2023-2024) con solo 37 secuencias causa overfitting severo. Entrenar en todo 2014-2024 combinado (192 seqs) produce mejor generalizacion. Ver Notebook 03 para ablacion.

# TravelIQ

Nationwide context-aware travel itinerary optimization system.

## Project Structure

```
traveliq/
├── models/                        # saved model weights — put your files here
│   ├── best_model.pt              # DistilBERT review model weights
│   ├── traveliq_tokenizer/        # tokenizer files
│   └── sarimax_models.pkl         # crowd score SARIMAX models
│
├── modules/
│   ├── review.py                  # sentiment + complaint + embeddings
│   ├── crowd.py                   # crowd index from SARIMAX forecast
│   └── weather.py                 # weather suitability scorer
│
├── pipeline.py                    # connects all modules → feature vectors
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place your saved model files in the `models/` folder:
- `best_model.pt` — your trained DistilBERT weights
- `traveliq_tokenizer/` — tokenizer folder from `tokenizer.save_pretrained()`
- `sarimax_models.pkl` — your crowd score models

## Run

```bash
# Test individual modules
python modules/review.py
python modules/crowd.py
python modules/weather.py

# Test full pipeline
python pipeline.py
```

## Feature Vector Output

Each attraction produces a **771-dim feature vector**:
- `[0:768]`  — 768-dim DistilBERT review embedding
- `[768]`    — crowd index (0-1, lower = less crowded)
- `[769]`    — weather suitability (0-1, higher = better)
- `[770]`    — sentiment score (0-1, higher = more positive reviews)

This feeds directly into the XGBoost ranker (next step).

## Pipeline Overview

```
User Input (city, dates, travel style)
        ↓
┌───────────────────────────────┐
│  modules/review.py            │ → review embedding + sentiment
│  modules/crowd.py             │ → crowd index
│  modules/weather.py           │ → weather suitability
└───────────────────────────────┘
        ↓
   pipeline.py → 771-dim feature vector per attraction
        ↓
   XGBoost Ranker → utility score per attraction     [TODO]
        ↓
   OR-Tools Optimizer → final itinerary              [TODO]
```

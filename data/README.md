# Data

Trip and station data are obtained from [HSL Open Data](https://www.hsl.fi/en/hsl/open-data).

## Required structure

```
data/
├── trips/                    # OD trip CSVs by year/month
│   ├── od-trips-2016/
│   │   ├── 2016-05.csv
│   │   ├── 2016-06.csv
│   │   └── ...
│   ├── od-trips-2017/
│   └── ...
└── stations/                 # Station coordinates (one CSV)
    └── *.csv                 # Must have columns: Nimi, Name, x (lon), y (lat)
```

## Download

1. **Trips** — HSL provides OD trip data as monthly CSVs. Download for the seasons you need (e.g. 2016–2024).
2. **Stations** — Download the Helsinki and Espoo city bike station list (coordinates).

Place the files as above, then run `python scripts/prepare_data.py` from the project root.

## Output (after `prepare_data.py`)

- `merged/trips_merged.csv` — Full cleaned dataset
- `train/train.csv` — Trips before 2022
- `validation/validation.csv` — Trips in 2022
- `test/test.csv` — Trips from 2023 onward

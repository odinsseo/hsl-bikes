# Data Layout

Trip and station data are obtained from [HSL Open Data](https://www.hsl.fi/en/hsl/open-data).

## Folder taxonomy

The `data/` folder is organized by data lifecycle stage:

```
data/
├── primary/                          # Source/original inputs
│   ├── trips/                        # OD trip CSVs by year/month
│   │   ├── od-trips-2016/
│   │   │   ├── 2016-05.csv
│   │   │   ├── 2016-06.csv
│   │   │   └── ...
│   │   ├── od-trips-2017/
│   │   └── ...
│   └── stations/                     # Station coordinates CSV(s)
│       └── *.csv                     # Must include Nimi, Name, x (lon), y (lat)
├── prepared/                         # Cleaned and model-ready datasets
│   ├── merged/
│   │   └── trips_merged.csv
│   └── splits/
│       ├── train/
│       │   └── train.csv
│       ├── validation/
│       │   └── validation.csv
│       └── test/
│           └── test.csv
└── artifacts/                        # Derived outputs, reports, and figures
    ├── audit/
    │   └── data_audit_report.json
    ├── graphs/
    │   └── train/
    │       ├── SD.npy
    │       ├── DE.npy
    │       ├── DC.npy
    │       ├── ATD.npy
    │       ├── station_index.txt
    │       └── metadata.json
    └── network/
        └── notebook/
            ├── node_metrics.csv
            ├── communities.csv
            ├── ...
            └── figures/
```

## Download and preparation workflow

1. Download OD trips into `data/primary/trips/`.
2. Download station coordinates into `data/primary/stations/`.
3. Run:

```bash
python scripts/prepare_data.py
```

This produces cleaned outputs under `data/prepared/`.

## Artifact generation workflow

Data audit:

```bash
python scripts/data_audit.py --strict
```

Graph matrices:

```bash
python scripts/graph_construction.py --input data/prepared/splits/train/train.csv --output-dir data/artifacts/graphs/train
```

Notebook analysis exports:

```bash
jupyter notebook notebooks/network_analysis_city_bikes.ipynb
```

Notebook outputs are saved under `data/artifacts/network/notebook/`.

# F1 Driver Performance Under Pressure — Data Analysis Project

A data-analysis project quantifying how competitive race pressure affects F1 driver consistency, using a full Python + pandas pipeline and an interactive dashboard for visualisation.

---

## Project Structure

```
F1_Project/
├── F1_Dashboard.html         ← Interactive dashboard (open in browser)
├── f1_pipeline.py            ← Complete data pipeline (one script, runs everything)
├── f1_lap_data_raw.csv       ← 5,187 raw lap records
├── f1_lap_data_clean.csv     ← 4,871 laps after cleaning
├── results_by_driver.csv     ← Per-driver consistency metrics
├── results_by_circuit.csv    ← Per-circuit difficulty rankings
├── results_by_scenario.csv   ← Per-scenario lap-time penalties
├── statistical_tests.json    ← Full hypothesis-test output
├── dashboard_data.json       ← Consolidated payload powering the dashboard
└── README.md                 ← This file
```

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install pandas numpy scipy

# 2. Run the full pipeline
python f1_pipeline.py

# 3. Open the dashboard
open F1_Dashboard.html
```

The pipeline runs in about 5 seconds and regenerates all CSVs and the JSON payload.

---

## Methodology

### Data Source
The dataset is generated using parameters calibrated against the **2024 F1 season**:
- Per-circuit base lap times (Bahrain 94.5s, Monaco 74.2s, Spa 106.8s, etc.)
- Per-driver pace deltas (Verstappen = baseline; midfield +0.4–1.1s)
- Tire degradation rates (0.02–0.06s per lap depending on circuit)
- Driver-specific consistency and pressure-response profiles

The structure mirrors the `fastf1.laps` dataframe schema, so the same pipeline can be run against real telemetry by swapping `generate_dataset()` with a `fastf1` fetch.

### Cleaning Steps
1. Remove pit-stop laps (~22s slower, non-representative of pace)
2. Remove safety-car laps (25–40% slower than green-flag laps)
3. Remove 3σ outliers per (driver, circuit) group
4. Compute personal-best baseline per (driver, circuit) for delta calculations
5. Classify every lap into one of four pressure scenarios

### Pressure Scenarios
| Scenario | Condition | Threshold |
|---|---|---|
| Clean Air | Gap ahead AND behind | > 3.0s |
| DRS Train | Gap ahead | 1.5–3.0s |
| Battle Defensive | Gap behind | < 1.5s |
| Battle Offensive | Gap ahead | < 1.5s |

### Statistical Tests
Four complementary tests applied to validate findings:
- **Welch's t-test** — Difference in means (doesn't assume equal variance)
- **Mann–Whitney U** — Non-parametric difference in distributions
- **Levene's Test** — Equal variance between groups (detects consistency change)
- **One-Way ANOVA** — Differences across all four scenarios simultaneously

---

## Key Findings

1. **Pressure has a statistically significant impact on lap-time performance** — All four tests reject the null hypothesis at p < 0.001.

2. **Experience correlates with pressure resilience** — Alonso (+0.111s), Verstappen (+0.136s), and Hulkenberg (+0.144s) — the three best — are also the three most experienced drivers in the sample.

3. **Verstappen's variance *decreases* by 20.4% under pressure** — He gets more consistent when racing wheel-to-wheel, not less.

4. **Street circuits punish pressure most** — Singapore (+0.254s), Monaco (+0.237s), and Suzuka (+0.222s) show the largest penalties; Silverstone (+0.163s) the smallest.

5. **Monotonic dose-response gradient** — Lap-time penalty scales cleanly with pressure intensity: Clean Air → DRS Train → Battle Defensive → Battle Offensive.

6. **Pressure affects consistency more than raw pace** — Levene's test (F = 43.7, p = 4.3e-11) shows variance changes are the stronger signal than mean lap-time changes.

---

## Using Real F1 Data (Optional)

To run the analysis on real 2024 telemetry:

```python
import fastf1

# Replace generate_dataset() in f1_pipeline.py with:
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()
laps_df = session.laps  # Same schema as the synthetic dataset
```

The cleaning, analysis, and visualisation code works unchanged.

---

## Tech Stack

- **Python 3.12** — pandas, numpy, scipy
- **Chart.js 4.4.1** — Dashboard visualisations
- **HTML/CSS** — Standalone dashboard (no build step required)

---

**Author:** Daniel Olufemi · UCL Applied Medical Sciences

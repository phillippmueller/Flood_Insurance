# Flood Adaptation ABM — Household Decisions, Insurance & EAD

This repository contains an agent‑based model (ABM) of household flood‑risk decisions built with [Mesa](https://mesa.readthedocs.io/). For each structure (household), the model weighs elevation investments and insurance choices under different policy scenarios and computes **Expected Annual Damage (EAD)** and the chosen action. A small plotting utility maps results by structure/census tract.

> High level: households minimize a **prospect‑theory–weighted loss** that combines residual flood damages (after insurance and public risk reduction) with the annualized cost of elevating their home. Policies change what options and discounts are available.

---

## Contents

```
agent.py               # Household agent logic (decision model, prospect utility, choices)
functions.py           # Flood frequency, damage curves, insurance rate lookup, EAD & utility functions
model.py               # Mesa model: builds agents & collects results; assigns public risk reduction
parameters.py          # Tunable parameters (risk perception weights, utility/gamma, GEV-like params)
simulation.py          # Batch runner: iterates policy × coverage settings; writes CSV outputs
Results Plot.py        # Optional: maps + subplots for EAD, insurance type/coverage, elevation
model_chart.drawio     # Architecture/flow diagram (open in diagrams.net / draw.io)
```

Expected **input data** live in `data/` (not included here):
- `full_data_for_simulation.csv` — one row per structure with attributes (see schema below).
- `rate_table.csv` — NFIP‑like rate table with columns `zone, height, building, contents`.
Outputs are written to `data/` and plots to `plots/`.

---

## Installation

Tested with Python 3.11+. Create a virtual environment and install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install mesa numpy pandas numpy-financial scipy matplotlib geopandas shapely mapclassify
```

> Note: `geopandas` may require platform-specific system packages (GEOS/PROJ). See GeoPandas docs if install fails.

---

## Data requirements

### 1) `data/full_data_for_simulation.csv` (columns)
The model expects at least the following columns (additional columns are preserved through to results):

| column | type | notes |
|---|---|---|
| `structure_id` | str/int | unique id per structure (joins results) |
| `GEOID` | str/int | census tract id (used for targeting public risk reduction) |
| `mortgage` | str | e.g. `"Housing Units with a Mortgage"` |
| `income` | categorical | one of: `"Income Below $45,000"`, `"Households with Income $45,000 - $49,999"`, `"Households with Income $50,000 - $59,999"`, `"Households with Income $60,000 - $74,999"`, `"Households with Income $75,000 - $99,999"`, `"Households with Income $100,000 - $124,999"`, `"Households with Income $125,000 - $149,999"`, `"Households with Income $150,000 - $199,999"`, `"Households with Income $200,000 or more"` |
| `race` | categorical | `"Minority Population"` or other |
| `education` | int | education level encoded `0–24` (normalized internally) |
| `ownership` | categorical | `"Owner-Occupied Housing Units"` or other |
| `property_flood_zone` | str | e.g. `A`, `VE`, `VO` (insurance requirement if in A/V and mortgaged) |
| `flood_elevation_list` | str/list | list of flood levels (feet) aligned with `return_period_list` (see below); stored as Python-like list string, e.g. `"[3.2, 5.1, 6.7, 8.9, 10.3]"` |
| `property_height` | float | finished-floor height (feet) |
| `BFE` | float | Base Flood Elevation (feet) |
| `area` | float | building footprint area (e.g., ft²) used for elevation cost |
| `building_type` | str | `"residential"` or other supported type for damage curves |
| `house_value` | float | replacement value (USD) |
| `geometry` | WKT string | optional; used by plotting script (EPSG:4326 recommended) |

**Important:** `flood_elevation_list` **must be the same length and order** as `return_period_list` used in `simulation.py`, and represent the flood level (ft) for each return period. Internally, damage curves convert feet to meters for the polynomial loss function.

### 2) `data/rate_table.csv` (columns)
```
zone,height,building,contents
A,-1,0.45,0.20
A,0,0.60,0.25
VE,0,1.20,0.60
...
```
- `zone`: NFIP zone (`A`, `VE`, `VO`, etc.)
- `height`: elevation relative to BFE (ft) at/under which the rate applies
- `building`, `contents`: annual rate factors (per $100 coverage).

---

## How the model works (quick tour)

### Risk perception
Each household computes a scalar **risk perception** from normalized socio‑economic factors:
```
risk = (a·income + b·race + c·education + d·ownership + e·government) / (a+b+c+d+e)
```
Weights `a…e` are in `parameters.py`. A government term increases with tract‑level initial risk if that tract receives public risk reduction.

### Loss weighting (prospect theory)
For each return period `T`, the chance weight is:
```
π = [core^γ] / [core^γ + (1-core)^γ]^(1/γ) , where core = min(1, 10^(2·risk - 1) · 1/T)
```
Utility of a monetary loss `L` is `U(L) = L^β` with `β = expected_utility_parameter`.

### Damages & EAD
For each return period, residual damage per structure is:
```
damage = house_value · damage_curve( flood_height ) · (1 - public_risk_reduction)
flood_height = max(0, flood_level(ft) - house_elevation(ft))       # elevation reduces hazard
residual_loss = max(0, damage - insurance_coverage) + annual_elevation_cost
```
Prospect utility sums `π · U(residual_loss)` across return periods. **EAD** is computed by trapezoidal integration over the (damage, 1/return period) points.

### Choices
Agents choose the action that **minimizes** prospect utility among: elevation options and insurance bundles allowed by the policy scenario.

- Elevation options: `[0, 2, 4, 6, 8]` ft (annualized via a loan using `numpy_financial.pmt`).
- Insurance bundles:
  - **NFIP**: coverage `[60k, 150k, 250k]`, rates from `rate_table.csv`, minus community discount `CRS_rewards`.
  - **Private**: coverage `[60k, 250k, 500k]`, rates approximated as 3× NFIP (after CRS).  
- **Insurance requirement**: if zone ∈ `{A, VE, VO}` and unit is mortgaged, insurance is required; otherwise coverage `0` is also considered.

### Policy scenarios
- **`pre_FIRM`**: households evaluate elevation ∈ `[0,2,4,6,8]` ft with either NFIP or private insurance (or none if permitted).
- **`voucher`**: the program first estimates elevation to **BFE + 1 ft** (if below), then evaluates NFIP coverage bundles under a lower loan rate.

Additionally, the model targets the highest‑risk tracts:
- Compute initial EAD with no actions; sort tracts by mean EAD.
- Apply a public risk reduction factor `risk_reduction_percentage` to the **top** `covered_census_tracts` tracts (both set in `simulation.py`).

---

## Running a simulation

Edit the top of `simulation.py` as needed; defaults are:
```python
census_tract_number_list = [0, 10, 25, 50]          # how many top tracts to receive public risk reduction
policy_list = ["pre_FIRM", "voucher"]               # scenarios to run
return_period_list = [5.886, 13.734, 24.7212, 61.803, 200]  # must match flood_elevation_list length
CRS_rewards = 0.25                                  # community rating discount (25%)
risk_reduction_percentage = 0.25                    # 25% public risk reduction
```
Run:
```bash
python simulation.py
```

This writes one CSV per scenario/setting, e.g. `data/result_voucher_25.csv`.

---

## Outputs

Each result CSV (`data/result_{policy}_{covered}.csv`) merges original structure columns with the following **per‑agent** outputs (from the Mesa data collector):

| column | meaning |
|---|---|
| `EAD` | Expected Annual Damage **after** chosen action (USD/yr) |
| `insurance_type` | `"NFIP"`, `"private"`, or `"No insurance"` |
| `insurance_coverage` | Coverage selected (USD) |
| `elevation` | Chosen elevation above current (ft) |
| `damage_list` | Residual damages per return period (list) |

> Diagnostics: during the step, the agent also computes `EAD_no_action` and prints it; if you want it saved, add it to `model.py`’s `mesa.DataCollector(agent_reporters=…)`.

---

## Plotting (optional)

The script `Results Plot.py` makes a 2×2 subplot per run (EAD, insurance type, coverage, elevation) and saves them to `plots/`:

```bash
python "Results Plot.py"
```

It expects a WKT `geometry` column in the result CSV. Adjust classification/bins as needed.

---

## Calibration & parameters

Edit `parameters.py` to tune behavior:

- `location`, `scale`, `shape` — distribution parameters (used by flood frequency helpers if you extend the model).
- `a…e` — weights for income, race, education, ownership, government in risk perception.
- `expected_utility_parameter` (`β`) — curvature of utility for losses.
- `gamma` — probability weighting parameter in the prospect function.
- `M` — a large number used as an initial sentinel for utility comparisons.

Insurance rate logic lives in `functions.py` (`get_rate_NFIP`, `insurance_rate`). Elevation costs are in `functions.elevation_cost(area, elevation)` (piecewise linear in feet × area, annualized with a loan in the agent).

---

## File tree (minimal)

```
data/
  full_data_for_simulation.csv   # you provide
  rate_table.csv                 # you provide
plots/                           # created by the plotting script
agent.py
functions.py
model.py
parameters.py
simulation.py
Results Plot.py
model_chart.drawio
```

---

## Common issues & tips

- **Missing input files**: ensure both CSVs exist under `data/` with the columns described above.
- **List parsing**: `flood_elevation_list` is parsed with `ast.literal_eval` — make sure it’s a valid Python list string.
- **Units**: `property_height` & `BFE` in **feet**; damage curves convert feet→meters internally.
- **CRS/Geo issues**: `Results Plot.py` assumes WKT in EPSG:4326. Reproject as needed before plotting.
- **Determinism**: random choices are disabled by default (insurance requirement is rule‑based); set seeds if you add stochastic elements.

---

## License & citation

Add your preferred license and citation here. If you use this code in research or consulting deliverables, please include a reference to this repository and the authors.

---

## Acknowledgements

This model structure draws on common practices in flood‑risk ABMs (prospect theory loss weighting, NFIP/private insurance bundles, and EAD integration). See `model_chart.drawio` for an architecture diagram of the data flow and decisions.

Hej!
dette er min kode til mit speciale. mere info følger.

# Monte Carlo study: DML-DiD (Chang 2020) with Lasso first steps when sparsity fails

## Layout
| Path | Content |
|---|---|
| `src/dmldid.py` | estimator: first steps (BCCH plug-in Lasso / logit Lasso, post-Lasso refit), DML-DiD for repeated outcomes and repeated cross sections, oracle, Abadie plug-in, unadjusted DiD, OR-DiD, per-fit diagnostics |
| `src/dgp.py` | data-generating processes and bases (unchanged from the 10 Oct version) |
| `src/mc.py` | one replication, Lemma 4 and Prop. 10 diagnostics, parallel runs, summaries with Monte Carlo standard errors |
| `src/registry.py` | the designs D1-D8 as cell grids |
| `src/store.py` | results store (Parquet, one directory per cell, resumable, metadata per run) |
| `src/plots.py`, `src/analysis.ipynb` | figures and tables from stored results only |
| `run_designs.py` | runs a design in a fresh process and stores it |
| `tests/` | pre-flight tests (identities, bases, estimator behaviour, reproducibility, store) |
| `audit/` | numerical audit of the extreme D5 draws |
| `data/Bribes_Regression.dta` | Sequeira (2016) data as shipped in Chang's replication package |
| `legacy/` | superseded code and `simulations_vol5_outputs.ipynb`, the only record of the 10 Oct tables |

## Designs
D1 Chang's DGP and trend version · D2 density (decay) · D3 alignment · D4 which nuisance is dense ·
D5 reference category · D6 overlap · D7 centring · D8 Sequeira, repeated cross sections. Grids: `src/registry.py`.

## Workflow
```
pip install -r requirements.txt
python -m pytest -q tests                 # must pass before any long run
git commit -am "..."                      # stored rows record the commit; commit before the final run
python run_designs.py D2 --time           # time one replication of the most expensive cell
python run_designs.py D2                  # all cells, R = 500, 4 workers; rerun to resume
python run_designs.py D5 --dry-run        # stored progress per cell
```
Then open `src/analysis.ipynb`. Results live in `results/<design>/<cell>/` (`cell.json`, `runs.jsonl`,
`part-*.parquet`); nothing is overwritten, and replication `i` always uses `SeedSequence(seed, spawn_key=(i,))`.
Keep the laptop awake during long runs (Windows: Power Options, lid closing does nothing when plugged in).

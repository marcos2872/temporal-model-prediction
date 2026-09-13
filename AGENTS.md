# AGENTS.md — temporal-model

Time-series forecasting repo: CETESB water-quality data → baseline notebooks → per-experiment results. No `src/`, no tests, no CI. Notebooks + `resultados/` are the codebase.

## Environment

- `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt`
- Run things with `.venv/bin/python` or `.venv/bin/jupyter` (venv is gitignored; never commit it).
- Prophet needs CmdStan (~200 MB, auto-downloads on first fit, needs `g++`/`make`). Notebooks treat Prophet as **optional**: if CmdStan is missing they skip it and continue — preserve that behavior.
- How-to-run lives in `notebooks/README.md`; experiment protocol details in each `resultados/<exp>/README.md`.

## Data (`dados/`, see `dados/README.md`)

- CETESB CSVs are **not** plain CSVs: encoding `windows-1252`, separator `;`, decimal comma, row 1 is a CETESB header (skip it), dates are `dd/mm/aaaa hh:mm`, empty cell = missing (~18%). Copy the pandas snippet from `dados/README.md` — do not guess parsing.
- Known landmine: OD sensor dead `21/07/2026 01:10 → 06/08/2026 11:30` (16.4 days of NaN). Any OD experiment must use the clean segment or handle the gap explicitly.
- Data line 1 of each file declares validation status; post-22/08/2026 is provisional.

## Experiment protocol (locked — keep comparable)

- Univariate only (one variable per experiment). `L=8640` (30 d), `H=288` (1 d), 5-min step, temporal 70/15/15 split **no shuffle** + pure 10-day holdout with 10 daily origins. Interpolation max 24 steps (2 h).
- ARIMA(2,1,2) runs on an **hourly grid** (`L=720h`/`H=24h`, repeat ×12) for cost — do not run it at 5-min resolution.
- Rulers to beat: pH seasonal-naive MAE 0.0501 (rolling) / 0.0466 (holdout); OD 0.1525 / 0.1550.

## Naming and layout

- Notebooks: `notebooks/NN-<modelo>-<variavel>.ipynb`. Results: `resultados/NN-<modelo>-<variavel>/` with `README.md` (metrics table + embedded figs + reading), `metricas_*.csv`, `modelos/`, `figs/`. Index in `resultados/README.md`.
- New experiments reuse the same split/protocol; add their row to `resultados/README.md` and checkbox to main README §7.
- Rename notebooks only with `git mv`; update the tree + checklist in main README, the experiment README reproduce command, and the `resultados/README.md` row.
- Re-running a notebook with `nbconvert --inplace` **overwrites** its `resultados/<exp>/` artifacts — back the folder up first if the old run matters. Typical runtime 5–10 min.

## Docs rules

- Main README: new performance numbers only with a verifiable citation (author + table/page); formulas are canonical study formulations, not literal quotes — say so.
- Never reference gitignored dirs (`pdfs/`, `pdfs_texto/`, `reviews/`, `reports/`, `artigos/`, `.venv`) from committed docs — they don't go to GitHub. Cite `busca_bibliografica/` + DOI/arXiv links instead.

## Remote run (only if SSH MCP is active)

- Detect: `sshmcp_list_servers` shows a configured server (e.g. `temporal-remote`).
  If no server / MCP unavailable → ignore this section, run locally with `.venv`.
- Flow when active (code local, compute remote, transfer via MCP — no git on remote):
  1. Local: create/edit `notebooks/NN-*.ipynb`; extend `.gitignore` per exp for `*.pkl >100MB`.
  2. Remote via sshmcp: `git pull --ff-only` (deps only: `dados/`, checkpoints — anonymous read works)
     → ensure venv (`uv venv` / `pip install -r requirements.txt` if needed)
     → `sshmcp_upload_file` the notebook → `nbconvert --execute --inplace` in background
     (`nohup ... &`, poll with short `ps`/`tail` calls — the MCP channel times out on long runs).
  3. Local: `sshmcp_download_file` (executed notebook) + `sshmcp_download_directory`
     (`resultados/<exp>/`, absolute paths — `~` doesn't expand) → verify no nested dupes,
     metrics/figs complete, 0 error outputs → evaluate → write `resultados/<exp>/README.md`
     + index rows → `feat(model):` + `docs:` commits + push (all git happens locally).
- Never run heavy training locally when remote is available; never commit `.venv/` nor `*.pkl >100MB`.

## Git

- Conventional commits (`docs:` / `feat(model):` …). Never commit or push without explicit user confirmation; when in doubt propose split (docs vs experiment) via the question tool. Do not amend pushed commits.

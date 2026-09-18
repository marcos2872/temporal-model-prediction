# AGENTS.md — temporal-model

Time-series forecasting repo: CETESB water-quality data → baseline notebooks → per-experiment results. No `src/`, no tests, no CI. `univariavel/notebooks/` + `univariavel/resultados/` are the codebase; `univariavel/dados/` + `univariavel/app.py` belong to the univariate track only (multivariate has its own `multivariavel/dados/`).

## Environment

- `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt`
- Run things with `.venv/bin/python` or `.venv/bin/jupyter` (venv is gitignored; never commit it).
- Prophet needs CmdStan (~200 MB, auto-downloads on first fit, needs `g++`/`make`). Notebooks treat Prophet as **optional**: if CmdStan is missing they skip it and continue — preserve that behavior.
- How-to-run lives in `univariavel/COMO-RODAR.md` (uni) and `multivariavel/COMO-RODAR.md` (multi); notebook index in `univariavel/notebooks/README.md`; experiment protocol details in each `univariavel/resultados/<exp>/README.md` (uni) and `multivariavel/PLANO.md` + `multivariavel/resultados/M*/README.md` (multi). Versioned probe in `univariavel/benchmark-2025/` (see its README). Theory and references in `METODOLOGIA.md`.

## Data (`univariavel/dados/`, see `univariavel/dados/README.md`)

- Regime anual: treino = 2024 (`univariavel/dados/treino/`), benchmark = 2025 (`univariavel/dados/benchmark/`, intocado até a avaliação final). Ambos 100% validados, sem trecho provisório.
- CETESB CSVs are **not** plain CSVs: encoding `windows-1252`, separator `;`, decimal comma, row 1 is a CETESB header (skip it), dates are `dd/mm/aaaa hh:mm`, empty cell = missing. Copy the pandas snippet from `univariavel/dados/README.md` — do not guess parsing.
- Known landmines (2024): pH outages 16–18/jan (2,3 d), 29/abr–02/mai (2,7 d), **27/mai–13/jun (~17 d)** + micro-tails; any pH window overlapping NaN is dropped — report coverage per val slice. OD 2024 has only micro-outages.
- Val = v1: 4 slices of 10 days, one per season (19–28/abr, 20–29/jul, 15–24/set, 20–24/nov); v2: 5 slices (adds 13–22/dez) with purge/embargo. Windows assigned by end date. 2025 is NEVER touched by train/val/early-stopping/tuning.

## Experiment protocol (locked — keep comparable)

- Univariate only (one variable per experiment). v1: `L=8640` (30 d), `H=288` (1 d), 5-min step, interpolation max 24 steps (2 h). v2 (exps 10–18): `L=2304` (8 d) + purge/embargo, same `H`. Daily anchors at 23:55.
- ARIMA(2,1,2) runs on an **hourly grid** (`L=720h`/`H=24h`, repeat ×12) for cost — do not run it at 5-min resolution.
- Rulers (benchmark 2025 rolante, primário): v1 pH ens 0.0509 · OD ens 0.2107 (served by the API pending user decision) · v2 PatchTST pH 0.0465 · OD 0.2056 (crowned in exp 18). Treino-2024 val rulers: pH ens 0.0357 · OD ens 0.1325.
- Run ONE remote job at a time (12c/23GB OOMs fast); cap threads (`OMP/MKL/OpenBLAS_NUM_THREADS=4`) when sharing the box, uncapped when solo. Never `sleep` inside remote commands (MCP channel times out); poll with short `cat`/`ls` calls. `pkill -f` patterns must not match your own command line — use the `[.]` bracket trick.

## Naming and layout

- Layout: univariate track lives in `univariavel/`: data `univariavel/dados/` (treino 2024 + benchmark 2025, univariada), API `univariavel/app.py`, notebooks `univariavel/notebooks/00-baseline-ph.ipynb … 08-benchmark-2025.ipynb` (v1) + `10-v2-…18-v2-benchmark-2025.ipynb` (v2; `09-analises-pos-benchmark` = pós-benchmark sem treino), results `univariavel/resultados/<exp>/` with `README.md` (metrics table + embedded figs + reading), `metricas_*.csv`, `modelos/`, `figs/`, plus the versioned probe `univariavel/benchmark-2025/`. Index in `univariavel/resultados/README.md`. Binaries under `modelos/` are NOT committed — see "Model checkpoints" below.
- New experiments reuse the same split/protocol; add their row to `univariavel/resultados/README.md` and update the status/rulers table in the main README.
- Rename notebooks only with `git mv`; update the tree + status in main README, the experiment README reproduce command, and the `univariavel/resultados/README.md` row.
- Re-running a notebook with `nbconvert --inplace` **overwrites** its `univariavel/resultados/<exp>/` artifacts — back the folder up first if the old run matters. Typical runtime 5–10 min.

## Model checkpoints (GitHub Release — never in git)

- `univariavel/resultados/*/modelos/` binaries (`.pt`, `.pkl`, `.pkl.gz`, `prophet_*.json`, `arima*.pkl`) are gitignored; only the tiny `normalizacao.json` / `ensemble.json` stay tracked. Checkpoints are regenerable via notebooks 00–07 and downloadable with `bash scripts/baixar_modelos.sh [--dir DIR] [--tag TAG]` (uses `gh`, falls back to `curl`; verifies `SHA256SUMS.txt` and extracts at the repo root, preserving `univariavel/resultados/<exp>/modelos/` paths). NOTE: the `modelos-v1` Release predates the `univariavel/` move — its tarballs still carry `resultados/<exp>/modelos/` paths; re-tar with the new prefix (or move after extract) before use.
- Upload (new/updated checkpoints): one asset per experiment (`00-baseline-ph-modelos.tar.gz` … `07-ensemble-od-modelos.tar.gz`), each containing ONLY the gitignored binaries with relative paths (e.g. `univariavel/resultados/06-ensemble-ph/modelos/{dlinear_res_ph.pt,lgbm_steps.pkl.gz}`); the 120+ MB `.pkl` are excluded — the API reads the `.pkl.gz`. Generate `SHA256SUMS.txt` with **basenames** (`cd` into the asset dir before `sha256sum *.tar.gz > SHA256SUMS.txt` — absolute paths break `sha256sum -c` after download). Then `gh release create <tag> --title "..." --notes-file notes.md <assets> SHA256SUMS.txt`, or `gh release upload <tag> --clobber` for fixes. Biggest asset ~50 MB; GitHub per-file limit is 2 GB.
- Release notes must contain: provenance (HEAD SHA the checkpoints came from), rulers (benchmark 2025 pH ens 0.0509 · OD ens 0.2107; val pH 0.0357 · OD 0.1325), an asset→experiment→files table, `SHA256SUMS.txt` mention, the download command, and the note that `.pkl` >100 MB are excluded while small JSONs stay in git.
- Before committing any removal/upload: fresh `gh release download` into an empty dir + `sha256sum -c SHA256SUMS.txt` + extract + checksum-compare vs local + `.venv/bin/python -c "from univariavel.app import carrega; carrega('ph'); carrega('od')"` + end-to-end `bash scripts/baixar_modelos.sh --dir /tmp/...`.

## Docs rules

- Main README: new performance numbers only with a verifiable citation (author + table/page); formulas are canonical study formulations, not literal quotes — say so.
- Never reference gitignored dirs (`pdfs/`, `pdfs_texto/`, `reviews/`, `reports/`, `artigos/`, `.venv`) from committed docs — they don't go to GitHub. Cite `busca_bibliografica/` + DOI/arXiv links instead.

## Remote run (only if SSH MCP is active)

- Detect: `sshmcp_list_servers` shows a configured server (e.g. `temporal-remote`).
  If no server / MCP unavailable → ignore this section, run locally with `.venv`.
- Flow when active (code local, compute remote, transfer via MCP — no git on remote):
  1. Local: create/edit `univariavel/notebooks/<exp>.ipynb` (e.g. `00-baseline-ph`, `10-v2-baseline-ph`); checkpoints never go to git (see "Model checkpoints") — no `.gitignore` change needed for new exps unless new binary extensions appear.
  2. Remote prep via sshmcp (no git commands on remote): check work dir, venv
     (`uv venv` / `pip install -r requirements.txt` if needed) and deps (`univariavel/dados/`,
     checkpoints). Missing files → `sshmcp_upload_file` / `sshmcp_upload_directory`.
  3. `sshmcp_upload_file` the notebook (absolute paths — `~` doesn't expand) →
     `nbconvert --execute --inplace` in background (`nohup ... &`, poll with short
     `ps`/`tail` calls — the MCP channel times out on long runs).
  4. Local: `sshmcp_download_file` (executed notebook) + `sshmcp_download_directory`
     (`univariavel/resultados/<exp>/`) → verify no nested dupes, metrics/figs complete,
     0 error outputs → evaluate → write `univariavel/resultados/<exp>/README.md` (record remote
     host + work dir as provenance) + index rows → `feat(model):` + `docs:` commits
     + push (all git happens locally).
- Never run heavy training locally when remote is available; never commit `.venv/` nor `univariavel/resultados/*/modelos/` binaries (only `normalizacao.json` / `ensemble.json` stay tracked).

## Git

- Conventional commits (`docs:` / `feat(model):` …). Never commit or push without explicit user confirmation; when in doubt propose split (docs vs experiment) via the question tool. Do not amend pushed commits.

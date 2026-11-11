# Swarm Reader (SwarmAI)

**Ask questions about a PDF and watch an ant colony find the answer.**

Swarm Reader cuts a PDF into chunks, links them into a graph, and releases a colony of simulated ants (Ant Colony Optimisation, ACO) that walk the graph looking for the chunks most relevant to your question. Every decision the ants make is recorded, so you can pause the run, click an ant or a pheromone trail, and see the exact maths behind its choices. The result is a retrieval method you can *watch*, *inspect* and *benchmark*, not a black box.

> **What this is not.** ACO is not claimed to beat vector search. The built-in benchmark exists so you can measure whether it helps on your documents, and at what cost.

---

## Contents

- [Features](#features) · [Quick start](#quick-start) · [Using the app](#using-the-app)
- [How it works](#how-it-works) · [Architecture](#architecture) · [API reference](#api-reference)
- [Themes and design system](#themes-and-design-system) · [Configuration](#configuration)
- [Benchmark methodology](#benchmark-methodology) · [Sessions and security](#sessions-and-security) 
- [Troubleshooting](#troubleshooting) · [Known limitations](#known-limitations)

---

## Features

| Area | What you get |
|---|---|
| **Retrieval** | PDF text extraction (with OCR fallback), sentence-aware chunking, hybrid BM25 + embedding relevance, a chunk graph, and an ACO search over it |
| **Answers** | Evidence sentences with page references; a written answer from Gemini if configured, otherwise quoted sentences; **Evidence Consensus** score (see [what it means](#evidence-consensus)) |
| **3D workspace** | Three.js "galaxy" of chunks, animated ants, gold pheromone trails, orbit/zoom/focus camera, optional slow auto-rotation |
| **Explainable ACO** | Click any ant or edge. See candidates, τ, η, w, the formula with real numbers, and the probabilities **recorded at decision time** |
| **Time travel** | Pause, resume, step, rewind and scrub iterations; inspectors always show the frozen state |
| **Compare** | Compare two ants (paths, common/unique chunks, quality, pheromone contribution) or two edges (pheromone history, traversals, deposits, evaporation) |
| **Colony Monitor** | KPIs, pheromone distribution and four charts, in *current / cumulative / final-run* scopes, with a marker on the selected iteration |
| **Insights** | Document overview, why-these-chunks, ant/edge/iteration explanations, follow-up questions, benchmark interpretation. Built from recorded data; Gemini may only rephrase |
| **Benchmark** | BM25 vs semantic vs hybrid vs RRF vs hybrid + ACO, with median latency, Hit@5 and (with labels) Precision/Recall/MRR/nDCG |
| **Learn** | A "How Colony Works" page with equations, a diagram and interactive demos that never touch the live simulation |
| **UI system** | Dark (default) and light themes, four independently minimizable panels with a dock, responsive layout, keyboard and screen-reader support |

---

## Quick start

**Requirements:** Python 3.10+, a modern browser, internet access on first load (Three.js and the Inter / JetBrains Mono fonts come from CDNs; the page falls back to system fonts if the font CDN is unreachable).

```bash
git clone <your-repo-url> swarmai && cd swarmai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Open **http://localhost:5000**, drop in a PDF, and ask a question.

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). If `sentence-transformers` is not installed the app still works in **keyword-only** mode; the Input panel says which mode is active.

### Optional: Gemini (free key)

Gemini improves two things only: it widens your question into a hypothetical answer for retrieval, and it phrases the final answer and explanations. Everything works without it.

```bash
export GEMINI_API_KEY="your-key"          # create one at https://aistudio.google.com
export GEMINI_MODEL="gemini-flash-latest" # optional; any Flash / Flash-Lite model id
python server.py
```

A question uses 2 Gemini calls; each insight uses at most 1. After a failed call (rate limit, bad key) Gemini is skipped for 20 s and the local fallback is used. Free-tier limits and data-use terms are set by Google. Check them before using sensitive documents.

### Optional: OCR for scanned PDFs

Scanned pages need the [Tesseract](https://github.com/tesseract-ocr/tesseract) binary (`apt install tesseract-ocr` / `brew install tesseract`). Without it, pages with no extractable text are skipped and the upload response lists them as a warning.

---

## Using the app

1. **Upload** a PDF in the **Input** panel (limits: 25 MB, first 60 pages, 900 chunks; anything skipped is reported, never silently dropped).
2. **Ask** a question and press *Release the ants*. The **Answer** panel shows the question, a loading state, then the answer with clickable page citations, evidence rows and the Evidence Consensus ring.
3. **Watch the colony** in the 3D graph. Use the bottom timeline to pause, step, rewind or scrub. The *Speed* slider changes playback speed.
4. **Inspect.** Click an ant, a gold trail, or a chunk. The **Inspector** shows its details; ant inspectors include the full decision calculation. Press **+ Compare** on two ants or two edges to compare them.
5. **Explain.** Open **SwarmAI Insights** for explanations of the retrieval, the colony, the selected ant/edge, or the current iteration.
6. **Switch tabs.** Switching never resets the simulation, selection or typed text.

### Layout

**Tabs are destinations; panels are tools.**

| Tab | Purpose |
|---|---|
| **3D Workspace** | The graph plus four tool panels (default view) |
| **Activity** | Processing status, uploaded documents, previous queries, errors, event log (kept in the page; lost on reload) |
| **Benchmark** | Run and chart the five-method comparison |
| **How Colony Works** | Educational page with live demos |
| **Colony Monitor** | Statistics and charts for the current run |

The four workspace panels (**Input**, **Inspector**, **SwarmAI Insights**, **Answer**) can each be minimized with the **–** button and restored from the **dock** at the top of the workspace. All four can be minimized at once. Minimizing keeps their content, scroll area and selections, and never triggers a request. The graph re-centres in the freed space with a smooth transition. A green dot on a dock icon means there is something new inside (for example a finished answer while the Answer panel was minimized). On screens narrower than 900 px the panels become an accordion bottom sheet.

### Camera bar

Auto-rotate is **on by default and slow**. Any mouse drag, scroll or camera button switches it **off**, and it only restarts when you press *Auto-rotate* again. Also: zoom in/out, reset camera, focus chunk / ant / edge (enabled when something suitable is selected).

### Keyboard and accessibility

Tabs support ←/→; every icon-only button has an accessible name; focus rings are always visible; minimized panels are removed from tab order; loading and status text is announced; evidence rows and citations are keyboard-operable; `prefers-reduced-motion` disables animation and starts auto-rotation OFF; status is never conveyed by colour alone.

---

## How it works

```
PDF ──► text (+OCR) ──► chunks ──► BM25 index + embeddings ──► chunk graph
                                                                    │
question ──► relevance η (BM25 + cosine, optional Gemini HyDE) ──►  ACO colony
                                                                    │
                         hot chunks ──► best sentences ──► answer + evidence
```

1. **Ingest.** PyMuPDF extracts text (OCR fallback), repeated headers/footers are dropped, text is split into sentence-aware chunks.
2. **Graph.** Each chunk is a node. Edges join its **8 nearest chunks by meaning** and its **neighbours in reading order**. Edge weight *w* is the similarity (0.6 for reading-order links).
3. **Relevance (η).** `0.5 × BM25 + 0.5 × cosine similarity`, normalised to 0–1 with a 0.02 floor. If Gemini is configured, a short hypothetical answer is added to the query first (HyDE).
4. **Colony.** Each round releases **24 ants**. An ant starts at a chunk sampled in proportion to η<sup>β</sup>, then takes up to **6 steps** to unvisited neighbours chosen by the rule below. Up to **14 rounds** run.
5. **Reinforcement.** After each round the best quarter (**6 elite ants**, ranked by the sum of η along their path) lay pheromone on their edges. Every round **35 %** of all pheromone evaporates. Pheromone is clamped to **[0.1, 6]**.
6. **Convergence.** Stops when the five "hottest" chunks are unchanged for **3 rounds in a row** (checked from round 5), otherwise at round 14.
7. **Answer.** The best sentence in each of the hottest chunks becomes the evidence. Gemini (if configured) writes a short answer from it; otherwise sentences are quoted.

### The move rule

```
                 τ(i,j)^α · η(j)^β · w(i,j)
P(i → j)  =  ───────────────────────────────────      over unvisited neighbours k of i
              Σ_k  τ(i,k)^α · η(k)^β · w(i,k)
```

With probability **ε = 10 %** an ant ignores the formula and picks a random unvisited neighbour (exploration), so the effective probability is `(1−ε)·P + ε/n`. The Ant Inspector shows both, and states when a step was an exploration move.

| Parameter | Default | Meaning |
|---|---|---|
| α | 1 | weight of pheromone (colony memory) |
| β | 2 | weight of relevance |
| ρ | 0.35 | evaporation per round |
| ε | 0.10 | random-exploration probability |
| ants / steps / rounds | 24 / 6 / 14 | colony size, walk length, max rounds |
| neighbours (k-NN) | 8 | semantic edges per chunk |
| τ₀ / τ_min / τ_max | 1 / 0.1 / 6 | initial and clamped pheromone |

### Evidence Consensus

The percentage in the Answer panel is the share of final-round ants whose path passed through the chosen chunks, scaled by how relevant the evidence sentences are. **It measures agreement inside the search. It is not the probability that the answer is correct.** Only a labelled evaluation can tell you that (see the benchmark).

### Why decisions can be trusted in the inspectors

The colony records each decision *as it happens*: candidates, τ, η, w, numerators and the probabilities actually used. The inspectors read those records, never recompute from later pheromone. Tests check that recorded probabilities sum to 1, match the formula, and differ from the final pheromone values where they should.

---

## Architecture

```
swarmai/
├── server.py                # Flask app: ingest, retrieval, ACO + recording, trace API, insights, benchmark
├── static/index.html        # Single-file frontend: design tokens, panels, Three.js scene, inspectors, charts
├── requirements.txt         # flask, pymupdf, numpy, sentence-transformers
├── requirements-dev.txt     # pytest
└── tests/
    ├── conftest.py
    ├── test_app.py          # ingestion, sessions/isolation, Q&A, benchmark, ACO unit tests
    ├── test_trace.py        # decision recording, trace/stats endpoints, insights
    ├── test_theme.py        # design-system lint: tokens, contrast, no hard-coded colours, a11y names
    └── ui/                  # jsdom harness (needs the server running)
        ├── run.js           # simulation, inspectors, timeline, benchmark, insights
        ├── layout.js        # tabs, panels/dock, camera, comparison, monitor, demos
        ├── polish.js        # themes, answer panel, accessibility, performance sanity
        ├── sample.pdf
        └── package.json
```

Key backend pieces: `run_colony()` (Ant System with elitist deposit, MMAS-style bounds, optional recording), `store_run()` (per-session run store), `decision_view()` / `edge_view()` / `iter_stats()` (derived views over the recording), `compare()` (benchmark), `insights()` (local facts + optional Gemini rephrase with a numbers guard).

---

## API reference

All endpoints need the session cookie issued by `/api/ingest`. Errors return `{"error": "..."}` with a meaningful status (400 bad input, 404 missing/expired/not yours, 413 too large, 503 embedding failure).

| Method & path | Purpose |
|---|---|
| `POST /api/ingest` (multipart `pdf`) | Read a PDF. Returns `doc_id`, chunks with 3D positions, topics, warnings |
| `POST /api/ask` `{q, doc_id}` | Run the colony. Returns the trace, answer, evidence, `consensus`, `run_id`, `params` |
| `POST /api/compare` `{q, doc_id, expected?, relevant_pages?, reps?, seed?}` | Five-method benchmark |
| `POST /api/insights` `{kind, doc_id, run_id?, iter?, ant?, a?, b?}` | `overview` · `retrieval` · `aco` · `iteration` · `ant` · `edge` · `followups` · `benchmark` |
| `GET /api/trace/<run>/ant?iter=&ant=` | One ant: every decision with recorded τ, η, w, numerators, probabilities, reason |
| `GET /api/trace/<run>/edge?a=&b=` | One edge: per-iteration pheromone, deposits, evaporation, traversals, decisions that considered it |
| `GET /api/trace/<run>/iter/<k>` | Pheromone of every touched edge after iteration *k* |
| `GET /api/trace/<run>/stats` | Per-iteration statistics for the Colony Monitor |

---

## Themes and design system

The interface follows a restrained "scientific workbench" style: charcoal/navy surfaces, violet and cyan accents, Inter for text and JetBrains Mono for numbers, IDs and formulas.

- **Switching:** the sun/moon button in the header. Dark is the default; the choice is stored in `localStorage` (`swarm.theme`) and applied before first paint to avoid a flash. Panel layout is stored under `swarm.panels`.
- **One source of colour:** all colours are CSS variables defined once for each theme (`:root` / `:root[data-theme="light"]`). Components, SVG charts and generated markup use `var(--…)`; there are no hard-coded colours elsewhere.
- **Three.js scene:** WebGL cannot read CSS variables, so a numeric palette mirrors the tokens. Switching theme **recolours the existing scene in place** (additive glow in dark, normal blending in light) without rebuilding it. `test_theme.py` fails if the palette and tokens drift apart.
- **Contrast:** every text token pair meets WCAG AA (≥ 4.5:1) in both themes and graphic colours meet 3:1 (checked by test).
- **Adding a colour:** add the variable to **both** theme blocks, use it via `var()`. The lint test rejects literals.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Enables query expansion and fluent answers/explanations |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model id |
| `SWARM_SECRET` | random per process | Signs the session cookie. Set it to keep sessions across restarts |
| `SWARM_HTTPS` | unset | Set to `1` to mark cookies `Secure` when serving over HTTPS |

Tunable constants live at the top of `server.py` (`ANTS`, `ITERS`, `STEPS`, `ALPHA`, `BETA`, `RHO`, `KNN`, `EPS`, `MAX_PAGES`, `MAX_CHUNKS`, `MAX_DOCS`, `DOC_TTL`).

---

## Benchmark methodology

`POST /api/compare` ranks the top 5 chunks with BM25, semantic, hybrid 50/50, hybrid RRF and hybrid + ACO on identical chunks and scores.

- **Latency** is the **median over N runs** (default 5) of the ranking step only. ACO includes its colony run on the pre-built graph. Shared costs are reported separately: query scoring (per question), embedding and graph build (once per document).
- **ACO is stochastic**, so it runs N times with seeds `seed+0 … seed+N−1`. Hit@5 is the share of runs that hit, metrics are medians, and *stability* is the mean overlap with the first run.
- **Hit@5** needs an **expected phrase** (substring match in the top-5 chunk text); otherwise it shows N/A.
- **Precision / Recall / MRR / nDCG @5** need **relevant pages**; every chunk on those pages counts as relevant. Labels are page-level, so treat results as indicative.
- One question on one document is anecdotal. ACO is not assumed to win; read the numbers.

---

## Sessions and security

- Each browser gets a signed random session id. Documents and runs are stored **in server memory under a random id plus the owning session**; every endpoint checks ownership, and a wrong, expired or evicted id returns the same "no longer available" message.
- One active document per session (a new upload replaces it); 8 documents server-wide; 2-hour expiry; last 2 runs per session (10 overall).
- Cookies are `HttpOnly`, `SameSite=Lax`, and `Secure` when `SWARM_HTTPS=1`. Uploads are capped at 25 MB and validated (PDF header, corruption, encryption).
- Model-written text is escaped before display; Gemini output that introduces numbers not present in the recorded facts is discarded.
- **Not production-grade persistence:** state is in-process memory, so it is lost on restart and does not work across multiple workers. Put it behind a real store before deploying at scale.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "That document is no longer available" | Session expired, document replaced, or the server restarted (random secret per run). Upload again, or set `SWARM_SECRET` |
| Answers are quoted sentences, not prose | No `GEMINI_API_KEY`, or Gemini failed/rate-limited (it retries after 20 s) |
| "keyword only" in the stats row | `sentence-transformers` not installed or the model could not download |
| "No text found" | Scanned PDF without Tesseract installed |
| Blank 3D area / console error | Three.js CDN unreachable; check your connection |
| Fonts look generic | Google Fonts unreachable; the system-font fallback is used |
| Port 5000 busy | Edit the `app.run(port=5000)` line at the bottom of `server.py` |

---

## Known limitations

- In-memory sessions only (see above); Activity history lives in the page and is lost on reload; one document per session (no document picker).
- UI verified through jsdom only: **no real-browser, visual, overlap, touch or device testing, and no screenshots**.
- Relevance labels are page-level; the colony adds latency over plain ranking and may not improve quality on short documents.
- Ant ids are per iteration (`i3-a7` = ant 7 of iteration 3); selecting an ant keeps its slot when you change iteration.
- While paused mid-iteration the monitor's *completed journeys* follow the animation, while other figures are the recorded totals for that iteration (the page says so).
- Gemini rewriting has been tested with mocked responses and failure cases, not against the live service.
- Not built: integrated PDF viewer, 2D graph view, conversation history, pheromone heatmap, ACO parameter playground, user accounts.


"""Swarm Reader backend (v3): Ant Colony Optimisation over a chunk graph.
Install:  pip install flask pymupdf numpy sentence-transformers
Optional: set GEMINI_API_KEY (free key from aistudio.google.com) -> query expansion + fluent answer
Run:      python server.py   ->  http://localhost:5000
"""
import os, re, math, zlib, json, time, uuid, secrets, statistics, collections, urllib.request
import numpy as np
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory, session
from werkzeug.exceptions import HTTPException

try:
    from sentence_transformers import SentenceTransformer
    EMB = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("Embeddings unavailable, using lexical fallback:", e)
    EMB = None

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
# Sessions: signed cookie holding a random id. Set SWARM_SECRET to keep sessions across restarts;
# otherwise a random key is generated per process (sessions reset when the server restarts).
app.secret_key = os.environ.get("SWARM_SECRET") or os.urandom(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("SWARM_HTTPS") == "1")
DOC_TTL = 2 * 3600  # seconds a document stays in memory
MAX_PAGES, MAX_CHUNKS, CHUNK = 60, 900, 260
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")  # any Flash / Flash-Lite model id
ANTS, ITERS, STEPS, ALPHA, BETA, RHO, KNN = 24, 14, 6, 1.0, 2.0, .35, 8
TAU0, TAU_MIN, TAU_MAX = 1.0, .1, 6.0
STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with which who what when where how why do does did not but if than then so such can could would should about into over also".split())
tok = lambda s: [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]
DOCS = collections.OrderedDict()  # doc_id -> retrieval state; one entry per upload, newest 8 kept
MAX_DOCS, EPS = 8, .10  # EPS: chance an ant explores a random neighbour

BUL = re.compile(r"^([•▪●◦\-–*]|\d{1,2}[.)])\s+")
ABBR = re.compile(r"\b(e\.g|i\.e|etc|vs|Fig|No|Dr|Mr|Mrs|approx)\.")
IMPERATIVE = re.compile(r"^(Explain|Describe|Discuss|Define|List|Compare|Differentiate|Write|Give|Draw|Illustrate|State)\b")


class BM25:
    def __init__(s, docs, k1=1.5, b=.75):
        s.N, s.k1, s.b = len(docs), k1, b
        s.tf = [collections.Counter(d) for d in docs]
        s.len = np.array([len(d) for d in docs]) + 1e-9
        s.avg = s.len.mean() if s.N else 1
        df = collections.Counter(w for d in docs for w in set(d))
        s.idf = {w: math.log(1 + (s.N - n + .5) / (n + .5)) for w, n in df.items()}

    def score(s, q):
        out = np.zeros(s.N)
        for i, tf in enumerate(s.tf):
            for w in q:
                f = tf.get(w)
                if f:
                    out[i] += s.idf[w] * f * (s.k1 + 1) / (f + s.k1 * (1 - s.b + s.b * s.len[i] / s.avg))
        return out


def vec(texts):
    if EMB:
        return EMB.encode(texts, normalize_embeddings=True, batch_size=64)
    V = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in tok(t):
            V[i, zlib.crc32(w.encode()) % 256] += 1
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


# ---------- text cleaning ----------
def page_text(page):
    t = page.get_text()
    if len(t.strip()) < 20:  # scanned page -> OCR (needs the Tesseract binary)
        try:
            t = page.get_text(textpage=page.get_textpage_ocr(full=True))
        except Exception:
            pass
    return t


def drop_boilerplate(raw):
    """Remove headers/footers: lines that repeat (ignoring digits) on many pages."""
    norm = lambda l: re.sub(r"\d+", "#", l.strip().lower())
    cnt = collections.Counter(n for t in raw for n in {norm(l) for l in t.split("\n") if l.strip()})
    lim = max(3, .4 * len(raw)) if len(raw) >= 3 else 10 ** 9
    return ["\n".join(l for l in t.split("\n") if not l.strip() or cnt[norm(l)] < lim) for t in raw]


def reflow(text):
    """Join wrapped lines, turn bullets and short headings into sentence boundaries."""
    res, last = "", 0
    for ln in (l.strip() for l in text.split("\n")):
        if not ln: continue
        m = BUL.match(ln)
        if m: ln = ln[m.end():]
        if res:
            if m or (res[-1] not in ".!?:;," and last < 50):
                res = (res.rstrip(";,") + ("" if res.rstrip(";,")[-1] in ".!?:" else ".")) + " "
            else:
                res += " "
        res += ln; last = len(ln)
    return re.sub(r"\bPage \d+( of \d+)?\b", "", res)


def split_sents(t):
    t = ABBR.sub(lambda m: m.group(1) + "\x01", t)
    return [p.replace("\x01", ".").strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", t) if p.strip()]


def chunk_page(text):
    out, cur = [], ""
    for s in split_sents(reflow(text)):
        if cur and len(cur) + len(s) > CHUNK:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur: out.append(cur)
    return [c for c in out if len(c) > 30]


def sentences(t):
    """Candidate answer sentences: complete, not headings, not exam-style questions."""
    ok = [s for s in split_sents(t) if len(s) >= 35 and not s.endswith("?") and not IMPERATIVE.match(s)]
    return ok


def kw_score(q, s, B):
    st = set(tok(s)); tot = sum(B.idf.get(w, 1) for w in q) or 1
    return sum(B.idf.get(w, 1) for w in q if w in st) / tot


def lower_first(s):
    w = s.split(" ")[0]
    return s if (w.isupper() and len(w) > 1) or re.match(r"[A-Z][a-z]*[A-Z]", w) else s[0].lower() + s[1:]


def trim(s):
    return re.sub(r"\s+", " ", BUL.sub("", s)).strip().rstrip(".;:,")


# ---------- Gemini (optional, 2 calls per question) ----------
GEM_BLOCK = [0.0]  # after a failed call (rate limit, bad key) skip Gemini briefly instead of hammering it


def gemini(prompt, n=900):
    key = os.environ.get("GEMINI_API_KEY")
    if not key or time.time() < GEM_BLOCK[0]: return None
    try:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            data=json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                             "generationConfig": {"maxOutputTokens": n, "temperature": .2}}).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        r = json.load(urllib.request.urlopen(req, timeout=25))
        return r["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print("Gemini call failed (rate limit or bad key?), continuing without it:", e); GEM_BLOCK[0] = time.time() + 20
        return None


# ---------- the colony ----------
def build_graph(V):
    """Chunk graph: k nearest neighbours by meaning + reading-order neighbours. Edge weight = similarity."""
    N = len(V); sim = V @ V.T; np.fill_diagonal(sim, -1)
    nb = [dict() for _ in range(N)]
    for i in range(N):
        for j in np.argsort(-sim[i])[:KNN]:
            if int(j) != i: nb[i][int(j)] = max(float(sim[i, j]), .05)
        for j in (i - 1, i + 1):
            if 0 <= j < N: nb[i][j] = max(nb[i].get(j, 0), .6)
    for i in range(N):
        for j, w in list(nb[i].items()): nb[j].setdefault(i, w)
    return nb


ek = lambda a, b: (a, b) if a < b else (b, a)


def step_nums(cur, c, tau, eta, nb):
    """Numerators tau^alpha * (eta+.02)^beta * w for each candidate (all strictly positive)."""
    return np.array([tau[ek(cur, j)] ** ALPHA * (eta[j] + .02) ** BETA * nb[cur][j] for j in c])


def step_probs(cur, c, tau, eta, nb):
    """Normalised move probabilities from node `cur` to candidate nodes `c`."""
    w = step_nums(cur, c, tau, eta, nb); return w / w.sum()


PARAMS = dict(alpha=ALPHA, beta=BETA, rho=RHO, eps=EPS, ants=ANTS, steps=STEPS, tau0=TAU0, tau_min=TAU_MIN, tau_max=TAU_MAX)
ekey = lambda a, b: f"{min(a, b)}-{max(a, b)}"


def run_colony(eta, nb, rng, rec=None):
    """Classic Ant System with elitist deposit and MMAS-style pheromone bounds.
    Move rule: P(i->j) = tau_ij^a * eta_j^b * w_ij / sum_k(...) over unvisited neighbours; with prob. EPS the ant instead
    picks a random unvisited neighbour (exploration). If `rec` (a dict) is given, every decision is recorded as it happens:
    rec['dec'][(it,ant)] -> list of steps with candidates, tau, eta, w, numerators and the probabilities actually used;
    rec['ants'], rec['tau_hist'] (pheromone of every touched edge after each iteration) and rec['ev'] (per-edge events)."""
    N = len(eta); tau = collections.defaultdict(lambda: TAU0); heat = np.zeros(N)
    p0 = eta ** BETA; p0 /= p0.sum(); trace, prev, same, explored = [], None, 0, set()
    if rec is not None: rec.update(dec={}, ants={}, tau_hist=[], ev=collections.defaultdict(list))
    for it in range(ITERS):
        paths = []
        for ai in range(ANTS):
            cur = int(rng.choice(N, p=p0)); path = [cur]; seen = {cur}; steps = []
            for st in range(STEPS):
                c = [j for j in nb[cur] if j not in seen]
                if not c: break
                nums = step_nums(cur, c, tau, eta, nb); p = nums / nums.sum()
                explore = rng.random() < EPS
                k = int(rng.integers(len(c))) if explore else int(rng.choice(len(c), p=p))
                if rec is not None:
                    steps.append(dict(step=st + 1, frm=cur, to=c[k], mode="explore" if explore else "exploit", cands=c,
                                      tau=[float(tau[ek(cur, j)]) for j in c], eta=[float(eta[j]) + .02 for j in c],
                                      w=[float(nb[cur][j]) for j in c], num=[float(x) for x in nums], p=[float(x) for x in p]))
                    rec["ev"][ekey(cur, c[k])].append(("trav", it, ai, 0.0))
                cur = c[k]; path.append(cur); seen.add(cur)
            paths.append(path)
            if rec is not None: rec["dec"][(it, ai)] = dict(start_p=float(p0[path[0]]), steps=steps)
        qual = [float(sum(eta[p])) for p in paths]
        elite = sorted(range(ANTS), key=lambda a: -qual[a])[:ANTS // 4]
        for e in list(tau):                                                     # evaporation
            old = tau[e]; tau[e] = max(TAU_MIN, old * (1 - RHO))
            if rec is not None: rec["ev"][f"{e[0]}-{e[1]}"].append(("evap", it, -1, old - tau[e]))
        heat *= (1 - RHO)
        for a in elite:                                                         # deposit
            d = qual[a] / len(paths[a])
            for x, y in zip(paths[a], paths[a][1:]):
                old = tau[ek(x, y)]; tau[ek(x, y)] = min(TAU_MAX, old + d)
                if rec is not None: rec["ev"][ekey(x, y)].append(("dep", it, a, tau[ek(x, y)] - old))
            for x in paths[a]: heat[x] += d
        if rec is not None:
            for a in range(ANTS):
                rec["ants"][(it, a)] = dict(quality=qual[a], mean_rel=qual[a] / len(paths[a]), elite=a in elite, path=paths[a],
                                            deposit=qual[a] / len(paths[a]) if a in elite else 0.0)
            rec["tau_hist"].append({f"{e[0]}-{e[1]}": v for e, v in tau.items()})
        top = [int(i) for i in np.argsort(-heat)[:5]]
        if rec is not None: rec.setdefault("top", []).append(top)
        top_e = sorted(tau.items(), key=lambda kv: -kv[1])[:120]
        explored |= {n for p in paths for n in p}; tv = list(tau.values()) or [TAU0]
        trace.append(dict(ants=paths, elite=elite, top=top, evap=RHO, best_q=round(max(qual), 3),
                          avg_q=round(float(np.mean(qual)), 3), unique=len(explored),
                          tau_mean=round(float(np.mean(tv)), 3), tau_max=round(float(max(tv)), 3),
                          tau=[[a, b, round(v, 2)] for (a, b), v in top_e]))
        same = same + 1 if set(top) == prev else 0; prev = set(top)
        if it >= 4 and same >= 3: break                                         # converged
    if rec is not None: rec["heat"] = heat
    return trace, heat, same >= 3


def compose(kept):
    if not kept: return ""
    out = f"According to page {kept[0]['page']}, {lower_first(trim(kept[0]['text']))}."
    if len(kept) > 1: out += f" Page {kept[1]['page']} adds that {lower_first(trim(kept[1]['text']))}."
    return out


# ---------- run store + explanations ----------
RUNS = collections.OrderedDict(); MAX_RUNS, RUNS_PER_SESSION = 10, 2


def own_doc(did):
    d = DOCS.get(did)
    if d is not None and time.time() - d["t"] > DOC_TTL: DOCS.pop(did, None); d = None
    return d if d is not None and d["owner"] == session.get("sid") else None


def get_run(rid):
    r = RUNS.get(rid)
    return r if r is not None and r["owner"] == session.get("sid") and time.time() - r["t"] <= DOC_TTL else None


def store_run(doc_id, rec, extra):
    owner = session.get("sid"); rid = uuid.uuid4().hex[:12]
    mine = [k for k, v in RUNS.items() if v["owner"] == owner]
    for k in mine[:max(0, len(mine) - RUNS_PER_SESSION + 1)]: del RUNS[k]
    RUNS[rid] = dict(owner=owner, doc_id=doc_id, t=time.time(), rec=rec, **extra)
    while len(RUNS) > MAX_RUNS: RUNS.popitem(last=False)
    return rid


def decision_view(dec, chunk_pages):
    """One recorded decision plus derived quantities. Probabilities are the RECORDED ones, never recomputed from later pheromone."""
    n = len(dec["cands"]); eff = [(1 - EPS) * p + EPS / n for p in dec["p"]]; k = dec["cands"].index(dec["to"])
    fac = lambda arr, pw: np.array(arr) ** pw
    ft, fe, fw = fac(dec["tau"], ALPHA), fac(dec["eta"], BETA), np.array(dec["w"])
    ratio = {"pheromone": float(ft[k] / ft.mean()), "relevance": float(fe[k] / fe.mean()), "link weight": float(fw[k] / fw.mean())}
    order = sorted(range(n), key=lambda i: -dec["p"][i]); rank = order.index(k) + 1
    if dec["mode"] == "explore":
        why = (f"Random exploration move (probability {EPS:.0%} per step): the ant picked uniformly among {n} unvisited neighbours, "
               f"so this candidate had a 1/{n} = {1 / n:.1%} chance. The formula alone gave it {dec['p'][k]:.1%} (rank {rank} of {n}).")
    else:
        drv = max(ratio, key=ratio.get)
        why = (f"Chosen by the formula with probability {dec['p'][k]:.1%} (rank {rank} of {n}). Compared with the average candidate its "
               f"pheromone term was {ratio['pheromone']:.2f}x, relevance term {ratio['relevance']:.2f}x and link weight {ratio['link weight']:.2f}x; "
               f"the biggest advantage came from {drv}.")
    return dict(dec, effective=eff, sum_num=float(sum(dec["num"])), chosen_idx=k, ratio=ratio, rank=rank, reason=why,
                pages=[int(chunk_pages[j]) for j in dec["cands"]], alpha=ALPHA, beta=BETA, eps=EPS)


def edge_view(run, a, b):
    rec, V, nb, pages = run["rec"], run["V"], run["nb"], run["pages"]; key = ekey(a, b); n_it = len(rec["tau_hist"])
    ev = rec["ev"].get(key, []); hist = []
    for it in range(n_it):
        e = [x for x in ev if x[1] == it]
        hist.append(dict(iter=it, tau=rec["tau_hist"][it].get(key, TAU0), deposited=sum(x[3] for x in e if x[0] == "dep"),
                         evaporated=sum(x[3] for x in e if x[0] == "evap"), ants=[x[2] for x in e if x[0] == "trav"],
                         depositors=[x[2] for x in e if x[0] == "dep"]))
    cons = []  # every recorded decision in which this edge was one of the options
    for (it, ai), dd in rec["dec"].items():
        for st in dd["steps"]:
            for end, other in ((a, b), (b, a)):
                if st["frm"] == end and other in st["cands"]:
                    i = st["cands"].index(other)
                    cons.append(dict(iter=it, ant=ai, step=st["step"], frm=end, to=other, p=st["p"][i], tau=st["tau"][i], chosen=st["to"] == other))
    cons.sort(key=lambda c: (c["iter"], c["ant"], c["step"]))
    w = nb[a].get(b, nb[b].get(a)); semantic = float(V[a] @ V[b])
    kind = "reading-order neighbours" if abs(a - b) == 1 else "semantic neighbours"
    tot_dep = sum(h["deposited"] for h in hist); tot_trav = sum(len(h["ants"]) for h in hist)
    expl = (f"This edge links chunk {a} (page {int(pages[a])}) and chunk {b} (page {int(pages[b])}) as {kind}; similarity {semantic:.2f}, weight w={w:.2f}. "
            f"Across {n_it} iterations ants crossed it {tot_trav} time(s) and elite ants deposited {tot_dep:.3f} pheromone in total; "
            f"its pheromone ended at {hist[-1]['tau']:.3f} (start {TAU0}).")
    return dict(a=a, b=b, key=key, pages=[int(pages[a]), int(pages[b])], similarity=semantic, weight=w, kind=kind, initial=TAU0,
                history=hist, considered=cons[:80], n_considered=len(cons), explanation=expl, params=PARAMS)


def iter_stats(run):
    """Per-iteration statistics computed only from the recorded run."""
    rec = run["rec"]; n = len(rec["tau_hist"]); dep, evp = [0.0] * n, [0.0] * n; trv, exr, exl = [0] * n, [0] * n, [0] * n
    for lst in rec["ev"].values():
        for kind, it, ai, amt in lst:
            if kind == "dep": dep[it] += amt
            elif kind == "evap": evp[it] += amt
            else: trv[it] += 1
    for (it, ai), dd in rec["dec"].items():
        for st in dd["steps"]:
            if st["mode"] == "explore": exr[it] += 1
            else: exl[it] += 1
    out, seen = [], set()
    for it in range(n):
        ants = [rec["ants"][(it, a)] for a in range(PARAMS["ants"])]; q = [x["quality"] for x in ants]; b = int(np.argmax(q))
        th = list(rec["tau_hist"][it].values()) or [TAU0]
        for x in ants: seen |= set(x["path"])
        out.append(dict(iter=it, deposited=dep[it], evaporated=evp[it], traversals=trv[it], explore=exr[it], exploit=exl[it],
                        best_q=max(q), avg_q=float(np.mean(q)), best_ant=b, best_path=ants[b]["path"], explored=len(seen),
                        elite=[a for a, x in enumerate(ants) if x["elite"]], tau_total=float(sum(th)), tau_mean=float(np.mean(th)),
                        tau_max=float(max(th)), tau_edges=len(th), tau_bins=np.histogram(th, bins=8, range=(TAU_MIN, TAU_MAX))[0].tolist(),
                        top=rec["top"][it]))
    return out


def facts_retrieval(f):
    out = []
    for c in f["chunks"]:
        out.append(f"Page {c['page']} (chunk {c['cid']}): keyword score {c['bm']:.2f} (rank {c['bm_rank']}), semantic score {c['sem']:.2f} (rank {c['sem_rank']}), "
                   f"combined relevance {c['raw']:.2f}; without ACO it would rank {c['hyb_rank']} by hybrid score; pheromone heat {c['heat']:.0%}; "
                   f"{c['visits']} of {f['n_ants']} final-round ants passed through it.")
    return out


def numbers_ok(text, facts):
    ok = set(re.findall(r"\d+(?:\.\d+)?", facts)) | {"1", "2", "3"}
    return all(x in ok for x in re.findall(r"\d+(?:\.\d+)?", text))


def llm_rewrite(instruction, facts_text):
    """Optional Gemini phrasing of locally computed facts. Rejected if it introduces numbers that are not in the facts."""
    t = gemini(f"{instruction}\nUse ONLY the facts below, cite pages like (p.3), add no numbers that are not listed, do not speculate.\n\nFACTS:\n{facts_text}", 900)
    return t if t and numbers_ok(t, facts_text) else None


def local_overview(d):
    texts, B = d["texts"], d["B"]; tf = collections.Counter(w for t in texts for w in tok(t))
    topics = sorted(tf, key=lambda w: -tf[w] * B.idf[w])[:8]
    sents = [(i, x) for i, t in enumerate(texts) for x in sentences(t)]
    defs = [f"{trim(x)} (p.{int(d['pages'][i])})" for i, x in sents if re.search(r"\b(is defined as|refers to|is a|are a|is an)\b", x)][:5]
    eqs = [f"{x.strip()[:120]} (p.{int(d['pages'][i])})" for i, x in sents if re.search(r"[A-Za-z]\s*=\s*[\w(]", x)][:5]
    cent = (d["V"] @ d["V"].mean(0)); first = [int(i) for i in np.argsort(-cent)[:3]]
    summary = " ".join(f"{trim(sentences(texts[i])[0])}. (p.{int(d['pages'][i])})" for i in first if sentences(texts[i]))
    return dict(summary=summary, topics=topics, concepts=topics[:6], definitions=defs, equations=eqs,
                questions=[f"What is {t}?" for t in topics[:4]])


def answerable(d, qs):
    """Keep only questions whose best chunk relevance clears the same threshold the answer pipeline uses."""
    keep = []
    for q in qs:
        try:
            _, _, bm, sem, _ = scent(d, q, use_llm=False)
            if (.5 * bm + .5 * sem).max() >= .2: keep.append(q)
        except Exception: pass
    return keep


# ---------- routes ----------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        msg = "File too large (limit 25 MB)." if e.code == 413 else e.description
        return jsonify(error=msg), e.code
    print("Unhandled error:", repr(e))
    return jsonify(error=f"Server error: {type(e).__name__}. Check the server console."), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    f = request.files.get("pdf")
    if not f: return jsonify(error="No file uploaded."), 400
    data = f.read()
    if not data: return jsonify(error="The uploaded file is empty."), 400
    if not data.lstrip()[:5].startswith(b"%PDF"): return jsonify(error="That file is not a PDF."), 400
    try: doc = fitz.open(stream=data, filetype="pdf")
    except Exception: return jsonify(error="Could not open this PDF. It may be corrupted."), 400
    if doc.needs_pass: return jsonify(error="This PDF is password-protected."), 400
    if len(doc) == 0: return jsonify(error="This PDF has no pages."), 400
    warnings = []
    if len(doc) > MAX_PAGES: warnings.append(f"Only the first {MAX_PAGES} of {len(doc)} pages were read.")
    raw = drop_boilerplate([page_text(doc[i]) for i in range(min(len(doc), MAX_PAGES))])
    texts, pages, capped = [], [], False
    for pi, t in enumerate(raw):
        for c in chunk_page(t):
            if len(texts) < MAX_CHUNKS: texts.append(c); pages.append(pi + 1)
            else: capped = True
    if capped: warnings.append(f"Chunk limit ({MAX_CHUNKS}) reached: later text was not indexed.")
    if not texts:
        return jsonify(error="No text found. If this is a scanned PDF, install Tesseract for OCR."), 400
    empty = [i + 1 for i, t in enumerate(raw) if not t.strip()]
    if empty: warnings.append(f"{len(empty)} page(s) had no extractable text (blank, or scanned without Tesseract OCR): "
                              + ", ".join(map(str, empty[:10])) + ("…" if len(empty) > 10 else ""))
    toks = [tok(t) for t in texts]; B = BM25(toks); t0 = time.perf_counter()
    try: V = vec(texts)
    except Exception as e:
        print("Embedding failure:", repr(e)); return jsonify(error="The embedding model failed while reading this document."), 503
    embed_ms = (time.perf_counter() - t0) * 1000
    X = V - V.mean(0); U, Sg, _ = np.linalg.svd(X, full_matrices=False); P = (U * Sg)[:, :3]
    if P.shape[1] < 3: P = np.pad(P, ((0, 0), (0, 3 - P.shape[1])))
    P = P / (np.abs(P).max() + 1e-9) * 60 + np.random.normal(0, .8, P.shape)
    t0 = time.perf_counter(); nb = build_graph(V); graph_ms = (time.perf_counter() - t0) * 1000
    owner = session.setdefault("sid", secrets.token_urlsafe(24)); now = time.time()
    for k in [k for k, v in DOCS.items() if v["owner"] == owner or now - v["t"] > DOC_TTL]: del DOCS[k]  # one active doc per session
    did = uuid.uuid4().hex[:12]
    DOCS[did] = dict(owner=owner, t=now, texts=texts, pages=np.array(pages), B=B, V=V, nb=nb,
                     prep=dict(embed_ms=round(embed_ms, 1), graph_ms=round(graph_ms, 1)))
    while len(DOCS) > MAX_DOCS: DOCS.popitem(last=False)
    tf = collections.Counter(w for d in toks for w in d)
    topics = sorted(tf, key=lambda w: -tf[w] * B.idf[w])[:14]
    return jsonify(doc_id=did, warnings=warnings, topics=topics, semantic=EMB is not None, pages=len(doc),
                   edges=sum(map(len, DOCS[did]["nb"])) // 2, gemini=bool(os.environ.get("GEMINI_API_KEY")),
                   chunks=[dict(t=t, p=p, pos=P[i].round(2).tolist()) for i, (t, p) in enumerate(zip(texts, pages))])


def get_request():
    """Validate JSON, question and document ownership. Returns (doc, question, json, error_response)."""
    j = request.get_json(silent=True)
    if not isinstance(j, dict): return None, None, {}, (jsonify(error="Invalid request."), 400)
    did = j.get("doc_id")
    if not did: return None, None, j, (jsonify(error="No document is active. Upload a PDF first."), 400)
    d = DOCS.get(did)
    if d is not None and time.time() - d["t"] > DOC_TTL: DOCS.pop(did, None); d = None
    if d is None or d["owner"] != session.get("sid"):  # same message for missing / expired / not yours
        return None, None, j, (jsonify(error="That document is no longer available (expired or replaced). Please upload the PDF again."), 404)
    q = str(j.get("q") or "").strip()
    if not q: return None, None, j, (jsonify(error="Type a question first."), 400)
    return d, q, j, None


def scent(d, question, use_llm=True):
    """Per-chunk relevance signals for a question (the ants' heuristic)."""
    hyp = gemini(f"Write a 2-sentence factual passage that would answer this question. Plain text only.\n\nQuestion: {question}", 600) if use_llm else None
    q = list(dict.fromkeys(tok(question) + (tok(hyp) if hyp else [])))
    qv = vec([question])[0]
    if hyp: qv = qv + vec([hyp])[0]; qv = qv / (np.linalg.norm(qv) + 1e-9)
    bm = d["B"].score(q); bm = bm / (bm.max() or 1)
    return q, qv, bm, np.clip(d["V"] @ qv, 0, 1), hyp


def rank_metrics(ids, rel):
    """Precision/Recall/MRR/NDCG @len(ids) against a set of relevant chunk ids."""
    h = [1 if i in rel else 0 for i in ids]
    dcg = sum(x / math.log2(k + 2) for k, x in enumerate(h)); ideal = sum(1 / math.log2(k + 2) for k in range(min(len(rel), len(ids))))
    return dict(precision=sum(h) / max(len(ids), 1), recall=sum(h) / max(len(rel), 1),
                mrr=next((1 / (k + 1) for k, x in enumerate(h) if x), 0.0), ndcg=dcg / ideal if ideal else 0.0)


@app.route("/api/compare", methods=["POST"])
def compare():
    """Benchmark five retrieval methods on one question.
    Shared work (query scoring, graph build) is reported separately. Per-method latency is the median over `reps`
    runs of the ranking step only; ACO is seeded (seed+rep) and also reports hit-rate / stability across runs."""
    d, question, j, err = get_request()
    if err: return err
    try: reps = min(15, max(1, int(j.get("reps", 5)))); seed = int(j.get("seed", 42))
    except (TypeError, ValueError): return jsonify(error="reps and seed must be integers."), 400
    exp = str(j.get("expected") or "").strip().lower(); texts, pages = d["texts"], d["pages"]
    rp = j.get("relevant_pages") or []
    if isinstance(rp, str): rp = [x for x in re.split(r"[,\s]+", rp) if x]
    try: relpages = {int(x) for x in rp}
    except (TypeError, ValueError): return jsonify(error="Relevant pages must be numbers like 2,5."), 400
    rel = {i for i, p in enumerate(pages) if int(p) in relpages} if relpages else None
    if rel is not None and not rel: return jsonify(error="None of those pages exist in this document."), 400
    t0 = time.perf_counter()
    try: q, qv, bm, sem, _ = scent(d, question, use_llm=False)
    except Exception as e: print("scent failed:", repr(e)); return jsonify(error="Retrieval failed (embedding model error)."), 503
    shared_ms = (time.perf_counter() - t0) * 1000; K = 5
    rk = lambda x: np.argsort(np.argsort(-x)); top = lambda x: [int(i) for i in np.argsort(-x)[:K]]
    hybrid = .5 * bm + .5 * sem; eta = np.clip(hybrid / (hybrid.max() or 1), .02, 1)
    methods = [("BM25 only", lambda r: top(bm), False), ("Semantic only", lambda r: top(sem), False),
               ("Hybrid 50/50", lambda r: top(hybrid), False),
               ("Hybrid RRF", lambda r: top(1 / (60 + rk(bm)) + 1 / (60 + rk(sem))), False),
               ("Hybrid + ACO", lambda r: top(run_colony(eta, d["nb"], np.random.default_rng(seed + r))[1]), True)]
    out = []
    for name, fn, stoch in methods:
        runs, ms = [], []
        for r in range(reps):
            t0 = time.perf_counter(); ids = fn(r); ms.append((time.perf_counter() - t0) * 1000); runs.append(ids)
        row = dict(method=name, stochastic=stoch, median_ms=round(statistics.median(ms), 3), ids=runs[0],
                   pages=sorted({int(pages[i]) for i in runs[0]}),
                   chunks=[dict(id=i, page=int(pages[i]), text=texts[i][:220]) for i in runs[0]],
                   hit_rate=round(sum(any(exp in texts[i].lower() for i in ids) for ids in runs) / reps, 3) if exp else None,
                   stability=round(float(np.mean([len(set(runs[0]) & set(x)) / len(set(runs[0]) | set(x)) for x in runs])), 3))
        if rel is not None:
            ms_ = [rank_metrics(ids, rel) for ids in runs]
            row["metrics"] = {k: round(statistics.median(m[k] for m in ms_), 3) for k in ms_[0]}
        out.append(row)
    d["bench"] = dict(results=out, k=K, reps=reps, question=question, has_expected=bool(exp), has_labels=rel is not None, t=time.time())
    d.get("ins", {}).clear()
    return jsonify(results=out, k=K, reps=reps, seed=seed, has_expected=bool(exp), has_labels=rel is not None,
                   preprocessing=dict(query_scoring_ms=round(shared_ms, 2), **d["prep"]))


@app.route("/api/ask", methods=["POST"])
def ask():
    d, question, _, err = get_request()
    if err: return err
    B, V, pages, texts, nb = d["B"], d["V"], d["pages"], d["texts"], d["nb"]
    # 1. scent: relevance of every chunk (optionally widened by a Gemini-written hypothetical answer, "HyDE")
    try: q, qv, bm, sem, hyp = scent(d, question)
    except Exception as e: print("scent failed:", repr(e)); return jsonify(error="Retrieval failed (embedding model error)."), 503
    raw = .5 * bm + .5 * sem
    if raw.max() < .2: return jsonify(iters=[], hits=[], answer=[], consensus=0, summary="", summary_mode="none", scored=len(texts), n_ants=ANTS, tau_max=TAU_MAX, expanded=bool(hyp), converged=False)
    eta = np.clip(raw / raw.max(), .02, 1)
    # 2. the colony explores the chunk graph
    rec = {}; trace, heat, converged = run_colony(eta, nb, np.random.default_rng(), rec)
    # 3. read out: best sentence in each hottest chunk, ranked by pheromone + sentence relevance
    hn = heat / (heat.max() or 1); cands = []
    for i in np.argsort(-heat)[:6]:
        ss = sentences(texts[int(i)])
        if not ss: continue
        sv = vec(ss); sc = [.5 * kw_score(q, s, B) + .5 * max(0, float(v @ qv)) for s, v in zip(ss, sv)]
        b = int(np.argmax(sc)); cands.append(dict(cid=int(i), page=int(pages[i]), text=trim(ss[b]) + ".", heat=round(float(hn[i]), 3),
                                                  sent=round(float(sc[b]), 3), rank=.6 * float(hn[i]) + .4 * float(sc[b])))
    kept, seen = [], set()
    for c in sorted(cands, key=lambda c: -c["rank"]):
        if c["text"] not in seen and len(kept) < 3: seen.add(c["text"]); kept.append(c)
    if kept: best = max(c["sent"] for c in kept); kept = [c for c in kept if c["sent"] >= .5 * best]  # drop weak tag-alongs
    last = trace[-1]["ants"]; ids = {c["cid"] for c in kept}
    for c in kept: c["visits"] = sum(c["cid"] in p for p in last)
    agree = sum(any(n in ids for n in p) for p in last) / len(last)
    conf = round(100 * min(.99, agree * (.4 + .6 * min(1, float(np.mean([c["sent"] for c in kept])) / .5)))) if kept else 0
    # 4. fluent answer (Gemini) or extractive fallback
    ev = "\n".join(f"[p.{c['page']}] {texts[c['cid']]}" for c in kept)
    summary = gemini("Answer the question in 1 to 3 plain sentences using ONLY the evidence below. Cite pages like (p.6). "
                     f"If the evidence does not answer it, say so.\n\nQuestion: {question}\n\nEvidence:\n{ev}", 900) if kept else None
    mode = "gemini"
    if not summary: summary, mode = compose(kept), "extractive"
    rk = lambda x: np.argsort(np.argsort(-x)) + 1; hyb = rk(raw); bmr, smr = rk(bm), rk(sem)
    fchunks = [dict(cid=c["cid"], page=c["page"], bm=float(bm[c["cid"]]), sem=float(sem[c["cid"]]), raw=float(raw[c["cid"]]), heat=c["heat"],
                    visits=c["visits"], bm_rank=int(bmr[c["cid"]]), sem_rank=int(smr[c["cid"]]), hyb_rank=int(hyb[c["cid"]])) for c in kept]
    rid = store_run(request.get_json()["doc_id"], rec, dict(V=V, nb=nb, pages=pages, question=question, converged=converged, n_iters=len(trace),
                    facts=dict(chunks=fchunks, n_ants=ANTS, converged=converged, iterations=len(trace), expanded=bool(hyp))))
    return jsonify(run_id=rid, params=PARAMS, iters=trace, hits=[int(i) for i in np.argsort(-eta)[:24]], answer=kept, consensus=conf, summary=summary,
                   summary_mode=mode, scored=len(texts), n_ants=ANTS, tau_max=TAU_MAX, expanded=bool(hyp), converged=converged)


def run_or_404(rid):
    r = get_run(rid)
    return r, (None if r else (jsonify(error="That simulation is no longer available. Ask the question again."), 404))


@app.route("/api/trace/<rid>/ant")
def trace_ant(rid):
    r, err = run_or_404(rid)
    if err: return err
    try: it, ai = int(request.args["iter"]), int(request.args["ant"])
    except (KeyError, ValueError): return jsonify(error="iter and ant must be integers."), 400
    rec = r["rec"]
    if (it, ai) not in rec["dec"]: return jsonify(error="No such ant."), 404
    dd, info = rec["dec"][(it, ai)], rec["ants"][(it, ai)]
    return jsonify(id=f"i{it + 1}-a{ai + 1}", iter=it, ant=ai, start_p=dd["start_p"], params=PARAMS, **info,
                   steps=[decision_view(st, r["pages"]) for st in dd["steps"]])


@app.route("/api/trace/<rid>/edge")
def trace_edge(rid):
    r, err = run_or_404(rid)
    if err: return err
    try: a, b = int(request.args["a"]), int(request.args["b"])
    except (KeyError, ValueError): return jsonify(error="a and b must be integers."), 400
    n = len(r["pages"])
    if not (0 <= a < n and 0 <= b < n) or a == b or (b not in r["nb"][a]): return jsonify(error="No such edge."), 404
    return jsonify(edge_view(r, a, b))


@app.route("/api/trace/<rid>/stats")
def trace_stats(rid):
    r, err = run_or_404(rid)
    if err: return err
    S = iter_stats(r)
    return jsonify(per_iter=S, params=PARAMS, converged=r["converged"], best=max(S, key=lambda x: x["best_q"])["iter"])


@app.route("/api/trace/<rid>/iter/<int:k>")
def trace_iter(rid, k):
    r, err = run_or_404(rid)
    if err: return err
    hist = r["rec"]["tau_hist"]
    if not 0 <= k < len(hist): return jsonify(error="No such iteration."), 404
    return jsonify(iter=k, n_iters=len(hist), converged=r["converged"], tau={e: round(v, 4) for e, v in hist[k].items()})


@app.route("/api/insights", methods=["POST"])
def insights():
    """Grounded explanations. Facts are computed locally from recorded data; Gemini (optional) only rephrases them and is
    discarded if it adds numbers that are not in the facts. Results are cached per document."""
    j = request.get_json(silent=True)
    if not isinstance(j, dict): return jsonify(error="Invalid request."), 400
    d = own_doc(j.get("doc_id")); kind = j.get("kind")
    if d is None: return jsonify(error="That document is no longer available. Please upload the PDF again."), 404
    cache = d.setdefault("ins", {}); run = get_run(j.get("run_id")) if j.get("run_id") else None
    key = f"{kind}|{j.get('run_id')}|{j.get('iter')}|{j.get('ant')}|{j.get('a')}-{j.get('b')}|{d.get('bench', {}).get('t')}"
    if key in cache: return jsonify(dict(cache[key], cached=True))
    gem = bool(os.environ.get("GEMINI_API_KEY")); res = None
    if kind == "overview":
        loc = local_overview(d); res = dict(kind=kind, data=loc, source="local")
        if gem:
            ctx = "\n".join(f"[p.{int(d['pages'][i])}] {d['texts'][i][:600]}" for i in np.linspace(0, len(d["texts"]) - 1, min(14, len(d["texts"]))).astype(int))
            t = gemini('Return ONLY JSON with keys summary (3 sentences, cite pages like (p.2)), topics (list), concepts (list), definitions (list of strings with page), '
                       'equations (list, only if present in the text), questions (4 questions answerable from the text).\n\nTEXT:\n' + ctx, 1800)
            try:
                g = json.loads(re.sub(r"^```(?:json)?|```$", "", (t or "").strip()).strip())
                if isinstance(g, dict) and g.get("summary"):
                    g["questions"] = answerable(d, [str(x) for x in g.get("questions", [])]) or loc["questions"]
                    res = dict(kind=kind, data={**loc, **{k: g[k] for k in loc if k in g}}, source="gemini")
            except Exception: pass
    elif kind in ("retrieval", "aco", "ant"):
        if run is None: return jsonify(error="Ask a question first, then request an explanation."), 404
        f, rec = run["facts"], run["rec"]
        if kind == "retrieval":
            lines = facts_retrieval(f); head = "Why these chunks were chosen:"
        elif kind == "aco":
            th = rec["tau_hist"]; ex = sum(1 for dd in rec["dec"].values() for st in dd["steps"] if st["mode"] == "explore"); tot = sum(len(dd["steps"]) for dd in rec["dec"].values())
            top = max(th[-1], key=th[-1].get); ev = rec["ev"][top]; a_, b_ = map(int, top.split("-"))
            m0 = float(np.mean(list(th[0].values()))); m1 = float(np.mean(list(th[-1].values())))
            lines = [f"The colony ran {f['iterations']} iteration(s) of {f['n_ants']} ants and {'converged (top-5 chunks unchanged for 3 rounds)' if f['converged'] else 'stopped at the round limit without converging'}.",
                     f"Move rule: probability proportional to pheromone^{ALPHA:g} x relevance^{BETA:g} x link weight; {EPS:.0%} of steps are random exploration ({ex} of {tot} recorded steps were).",
                     f"Evaporation removes {RHO:.0%} of pheromone each round, so unused trails fade and ants keep exploring; mean pheromone went from {m0:.2f} to {m1:.2f}.",
                     f"Strongest edge: chunk {a_} (p.{int(run['pages'][a_])}) - chunk {b_} (p.{int(run['pages'][b_])}), pheromone {th[-1][top]:.2f}, crossed {sum(1 for x in ev if x[0] == 'trav')} time(s), "
                     f"{sum(x[3] for x in ev if x[0] == 'dep'):.2f} deposited by elite ants."]
            head = "How the colony behaved:"
        else:
            try: it, ai = int(j["iter"]), int(j["ant"]); dd = rec["dec"][(it, ai)]
            except (KeyError, ValueError, TypeError): return jsonify(error="No such ant."), 404
            lines = [f"Step {st['step']}: {decision_view(st, run['pages'])['reason']}" for st in dd["steps"]] or ["This ant had no unvisited neighbours and stayed at its start chunk."]
            head = f"Ant i{it + 1}-a{ai + 1}:"
        facts_text = "\n".join(lines); txt = None
        if gem: txt = llm_rewrite("Explain this clearly for a non-expert in 3-5 sentences.", facts_text)
        res = dict(kind=kind, text=txt or f"{head}\n" + facts_text, facts=lines, source="gemini (numbers checked)" if txt else "local")
    elif kind == "iteration":
        if run is None: return jsonify(error="Ask a question first, then request an explanation."), 404
        S = iter_stats(run)
        try: it = int(j["iter"]); s_ = S[it]; assert it >= 0
        except (KeyError, ValueError, TypeError, IndexError, AssertionError): return jsonify(error="No such iteration."), 404
        pg = run["pages"]; prev = S[it - 1] if it else None; pgs = sorted({int(pg[c]) for c in s_["best_path"]})
        stable = 0
        for i in range(1, it + 1): stable = stable + 1 if set(S[i]["top"]) == set(S[i - 1]["top"]) else 0
        lines = [f"Iteration {it + 1} of {len(S)}: {PARAMS['ants']} ants released, {s_['traversals']} edge traversals, {s_['explored']} distinct chunks explored so far.",
                 f"Best path quality {s_['best_q']:.2f} (ant a{s_['best_ant'] + 1}, pages {', '.join(map(str, pgs))}); average {s_['avg_q']:.2f}"
                 + (f"; previous iteration: best {prev['best_q']:.2f}, average {prev['avg_q']:.2f}." if prev else "."),
                 f"Moves: {s_['exploit']} followed the probability rule and {s_['explore']} were random exploration steps.",
                 f"{len(s_['elite'])} elite ants deposited {s_['deposited']:.2f} pheromone; evaporation removed {s_['evaporated']:.2f} ({RHO:.0%} per round).",
                 f"Pheromone over {s_['tau_edges']} touched edges: mean {s_['tau_mean']:.2f}, max {s_['tau_max']:.2f}.",
                 "Hottest chunks: " + ", ".join(f"chunk {c} (p.{int(pg[c])})" for c in s_["top"][:3])
                 + (f"; same top-5 as the previous iteration ({stable} consecutive stable iteration(s), 3 needed after iteration 5)." if prev and set(prev["top"]) == set(s_["top"]) else "; the top-5 changed versus the previous iteration." if prev else ".")]
        facts_text = "\n".join(lines); txt = llm_rewrite("Explain what happened in this iteration of the ant colony in 3-5 sentences.", facts_text) if gem else None
        res = dict(kind=kind, text=txt or facts_text, facts=lines, source="gemini (numbers checked)" if txt else "local")
    elif kind == "edge":
        if run is None: return jsonify(error="Ask a question first, then request an explanation."), 404
        try:
            a_, b_ = int(j["a"]), int(j["b"])
            if b_ not in run["nb"][a_] or a_ == b_: raise KeyError
            ev_ = edge_view(run, a_, b_)
        except (KeyError, ValueError, TypeError, IndexError): return jsonify(error="No such edge."), 404
        taken = sum(c["chosen"] for c in ev_["considered"])
        lines = [ev_["explanation"], "Pheromone after each iteration: " + ", ".join(f"{h['tau']:.2f}" for h in ev_["history"]) + ".",
                 f"Considered in {ev_['n_considered']} recorded decision(s); taken in {taken} of the first {len(ev_['considered'])} listed.",
                 f"Evaporation removed {sum(h['evaporated'] for h in ev_['history']):.3f} in total and deposits added {sum(h['deposited'] for h in ev_['history']):.3f}."]
        facts_text = "\n".join(lines); txt = llm_rewrite("Explain why this edge has the pheromone it has, in 3-4 sentences, for a non-expert.", facts_text) if gem else None
        res = dict(kind=kind, text=txt or facts_text, facts=lines, source="gemini (numbers checked)" if txt else "local")
    elif kind == "followups":
        loc = local_overview(d)["questions"]; extra = []
        if run is not None and run["facts"]["chunks"]:
            c0 = run["facts"]["chunks"][0]["cid"]; tf = collections.Counter(tok(d["texts"][c0]))
            extra = [f"What does the document say about {w}?" for w, _ in sorted(tf.items(), key=lambda kv: -kv[1] * d["B"].idf.get(kv[0], 1))[:2]]
        cand = extra + loc
        if gem:
            t = gemini("Suggest 4 short follow-up questions that the text below can answer. One per line, no numbering.\n\nTEXT:\n" +
                       "\n".join(d["texts"][c["cid"]][:500] for c in (run["facts"]["chunks"] if run else [])[:3]) or d["texts"][0][:800], 400)
            if t: cand = [re.sub(r"^[-*\d.\s]+", "", x).strip() for x in t.splitlines() if x.strip().endswith("?")] + cand
        qs = list(dict.fromkeys(answerable(d, cand)))[:4]
        res = dict(kind=kind, questions=qs, source="gemini + relevance check" if gem and qs else "local")
    elif kind == "benchmark":
        bm_ = d.get("bench")
        if not bm_: return jsonify(error="Run the benchmark first."), 404
        R = bm_["results"]; fast = min(R, key=lambda r: r["median_ms"]); aco = R[-1]; hyb = R[2]
        lines = [f"Question: {bm_['question']!r}; {bm_['reps']} run(s) per method, top {bm_['k']}.",
                 f"Fastest: {fast['method']} ({fast['median_ms']} ms median). Hybrid + ACO took {aco['median_ms']} ms versus {hyb['median_ms']} ms for Hybrid 50/50."]
        if bm_["has_expected"]: lines.append("Hit@5 (expected phrase found): " + "; ".join(f"{r['method']} {r['hit_rate']:.0%}" for r in R) + ".")
        else: lines.append("No expected phrase was given, so Hit@5 is not available.")
        if bm_["has_labels"]: lines.append("nDCG@5: " + "; ".join(f"{r['method']} {r['metrics']['ndcg']:.2f}" for r in R) + ".")
        else: lines.append("No relevant pages were labelled, so precision, recall, MRR and nDCG are not available.")
        lines.append(f"ACO stability across runs (overlap with the first run): {aco['stability']:.2f}.")
        lines.append("Limits: one question and one document; relevance labels are page-level; latency covers ranking only; ACO is stochastic so a single run can differ.")
        facts_text = "\n".join(lines); txt = llm_rewrite("Interpret these benchmark results in 4-6 sentences, including trade-offs and limitations.", facts_text) if gem else None
        res = dict(kind=kind, text=txt or facts_text, facts=lines, source="gemini (numbers checked)" if txt else "local")
    else: return jsonify(error="Unknown insight type."), 400
    res["gemini_available"] = gem; cache[key] = res
    return jsonify(dict(res, cached=False))


if __name__ == "__main__":
    app.run(port=5000)

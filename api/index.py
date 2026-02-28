"""
BloodstockAI — Vercel Serverless Backend
FastAPI + Neon Postgres · DLC processing via external GPU worker
"""
import os
import uuid
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloodstockai")

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
VALID_GAITS = {"walk", "trot", "canter", "gallop"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB (Vercel Pro limit)
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DLC_MODEL = os.environ.get("BAI_DLC_MODEL", "superanimal_quadruped")
DLC_BACKBONE = os.environ.get("BAI_DLC_BACKBONE", "hrnet_w32")


def _pg_url():
    url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL") or ""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def get_db():
    url = _pg_url()
    if not url:
        raise HTTPException(503, "Database not configured. Set POSTGRES_URL in Vercel → Settings → Environment Variables.")
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor,
                            connect_timeout=5)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '10s'")
    cur.close()
    return conn


# ═══════════════════════════════════════════════════════════════
# Schema — auto-creates on first call
# ═══════════════════════════════════════════════════════════════
_SCHEMA_INIT = False

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    filename TEXT,
    file_size INTEGER,
    gait TEXT,
    video_width INTEGER,
    video_height INTEGER,
    video_fps REAL,
    video_duration REAL,
    video_total_frames INTEGER,
    sampled_frames INTEGER,
    dlc_model TEXT,
    bai_score REAL,
    bai_band TEXT,
    pis INTEGER,
    pas INTEGER,
    cbs INTEGER,
    mvs INTEGER,
    conf_overall REAL,
    movement_overall REAL,
    risk_flags TEXT,
    results_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
"""


def ensure_schema():
    global _SCHEMA_INIT
    if _SCHEMA_INIT:
        return
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
        conn.commit()
        cur.close()
        conn.close()
        _SCHEMA_INIT = True
        logger.info("Database schema initialized")
    except Exception as e:
        logger.error(f"Schema init failed: {e}")
        raise


# ═══════════════════════════════════════════════════════════════
# Keypoint mapping (DLC 39 → BAI 22) — shared with worker
# ═══════════════════════════════════════════════════════════════
BAI_KPS = [
    "nose", "eye", "nearear", "farear", "poll", "throat",
    "withers", "back", "loin", "croup", "tailbase",
    "shoulder", "elbow", "foreknee", "forefetlock", "forehoof",
    "hip", "stifle", "hock", "hindfetlock", "hindhoof", "girth",
]
BAI_GRP = {
    "nose": "head", "eye": "head", "nearear": "head", "farear": "head",
    "poll": "head", "throat": "head", "withers": "top", "back": "top",
    "loin": "top", "croup": "top", "tailbase": "top", "shoulder": "fore",
    "elbow": "fore", "foreknee": "fore", "forefetlock": "fore", "forehoof": "fore",
    "hip": "hind", "stifle": "hind", "hock": "hind", "hindfetlock": "hind",
    "hindhoof": "hind", "girth": "barrel",
}
BAI_LABELS = {k: k.replace("_", " ").title() for k in BAI_KPS}
BAI_LABELS.update({
    "nearear": "Near Ear", "farear": "Far Ear", "forefetlock": "Fore Fetlock",
    "forehoof": "Fore Hoof", "foreknee": "Fore Knee", "hindfetlock": "Hind Fetlock",
    "hindhoof": "Hind Hoof", "tailbase": "Tail Base",
})


# ═══════════════════════════════════════════════════════════════
# Biometric calculations (pure math — no numpy/cv2)
# ═══════════════════════════════════════════════════════════════
def _ang(a, b, c):
    ax, ay = a[0] - b[0], a[1] - b[1]
    bx, by = c[0] - b[0], c[1] - b[1]
    dot = ax * bx + ay * by
    m1, m2 = math.hypot(ax, ay), math.hypot(bx, by)
    if m1 == 0 or m2 == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (m1 * m2)))))


def _dst(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _avg(a):
    return sum(a) / len(a) if a else 0.0


def _std(a):
    if not a:
        return 0.0
    m = _avg(a)
    return math.sqrt(sum((v - m) ** 2 for v in a) / len(a))


def _rom(a):
    return (max(a) - min(a)) if a else 0.0


def compute_conformation(kps):
    lk = {k["id"]: (k["x"], k["y"]) for k in kps if k.get("conf", 0) > 0.1}
    m = {}
    def has(*ids):
        return all(i in lk for i in ids)
    if has("nose", "poll", "withers"):
        m["headNeckAngle"] = _ang(lk["nose"], lk["poll"], lk["withers"])
    if has("withers", "shoulder", "elbow"):
        m["shoulderAngle"] = _ang(lk["withers"], lk["shoulder"], lk["elbow"])
    if has("croup", "hip", "stifle"):
        m["hipAngle"] = _ang(lk["croup"], lk["hip"], lk["stifle"])
    if has("stifle", "hock", "hindfetlock"):
        m["hockAngle"] = _ang(lk["stifle"], lk["hock"], lk["hindfetlock"])
    if has("elbow", "foreknee", "forefetlock"):
        m["foreKneeAngle"] = _ang(lk["elbow"], lk["foreknee"], lk["forefetlock"])
    if has("withers", "back", "loin", "croup"):
        m["toplineDeviation"] = ((lk["back"][1] + lk["loin"][1]) / 2 - (lk["withers"][1] + lk["croup"][1]) / 2) * 100
    if has("shoulder", "hip", "withers", "forehoof"):
        denom = _dst(lk["withers"], lk["forehoof"])
        m["bodyLengthRatio"] = _dst(lk["shoulder"], lk["hip"]) / denom if denom > 0 else 0
    if has("poll", "withers", "forehoof"):
        denom = _dst(lk["withers"], lk["forehoof"])
        m["neckLengthRatio"] = _dst(lk["poll"], lk["withers"]) / denom if denom > 0 else 0
    if has("croup", "tailbase"):
        m["croupAngle"] = abs(math.degrees(math.atan2(
            lk["tailbase"][1] - lk["croup"][1], lk["tailbase"][0] - lk["croup"][0])))
    return m


SCORE_DEFS = [
    ("shoulderAngle", 95, 30, 1.5, "Shoulder Angle", "\u224895\u00B0"),
    ("headNeckAngle", 110, 40, 1.0, "Head-Neck", "\u2248110\u00B0"),
    ("hipAngle", 95, 30, 1.5, "Hip Angle", "\u224895\u00B0"),
    ("hockAngle", 155, 30, 1.2, "Hock Angle", "\u2248155\u00B0"),
    ("foreKneeAngle", 170, 25, 1.2, "Fore Knee", "\u2248170\u00B0"),
    ("toplineDeviation", 0, 8, 1.3, "Topline Dev.", "\u22480"),
    ("bodyLengthRatio", 1.0, 0.35, 1.0, "Body Ratio", "\u22481.0"),
    ("neckLengthRatio", 0.38, 0.2, 0.8, "Neck Ratio", "\u22480.38"),
    ("croupAngle", 25, 20, 1.0, "Croup Angle", "\u224825\u00B0"),
]


def score_conformation(metrics):
    scores = {}
    ts = tm = 0
    for key, ideal, tol, weight, label, ideal_str in SCORE_DEFS:
        if key in metrics:
            raw = max(0, 1 - abs(metrics[key] - ideal) / tol)
            sc = raw * 10 * weight
            mx = 10 * weight
            scores[key] = {
                "score": round(sc, 2), "maxScore": round(mx, 2),
                "raw": round(raw, 3), "value": round(metrics[key], 2),
                "ideal": ideal_str, "label": label,
                "pct": round((sc / mx) * 100, 1) if mx > 0 else 0,
            }
            ts += sc
            tm += mx
    overall = (ts / tm * 100) if tm > 0 else 0
    return scores, round(overall, 1)


GAIT_SF = {"walk": 1.0, "trot": 1.4, "canter": 1.8, "gallop": 2.4}


def compute_movement(frame_kps, gait="trot"):
    if len(frame_kps) < 2:
        return {"overall": 0, "metrics": {}, "foreDeltas": [], "hindDeltas": [], "strides": 0}
    sf = GAIT_SF.get(gait, 1.4)
    fore_d, hind_d = [], []
    for i in range(1, len(frame_kps)):
        prev = {k["id"]: k for k in frame_kps[i - 1]}
        curr = {k["id"]: k for k in frame_kps[i]}
        for hoof_id, target in [("forehoof", fore_d), ("hindhoof", hind_d)]:
            if hoof_id in prev and hoof_id in curr:
                p, c = prev[hoof_id], curr[hoof_id]
                if p.get("conf", 0) > 0.1 and c.get("conf", 0) > 0.1:
                    target.append(math.hypot(c["x"] - p["x"], c["y"] - p["y"]))
    af, ah = _avg(fore_d), _avg(hind_d)
    symmetry = (min(af, ah) / max(af, ah) * 100) if af > 0 and ah > 0 else 0
    all_d = fore_d + hind_d
    rhythm = max(0, (1 - _std(all_d) / (_avg(all_d) or 1))) * 100
    per_frame = [compute_conformation(kps) for kps in frame_kps]
    tls = max(0, (1 - _std([m.get("toplineDeviation", 0) for m in per_frame]) / 3)) * 100
    hv = [m.get("headNeckAngle", 0) for m in per_frame if m.get("headNeckAngle")]
    hcs = max(0, (1 - _std(hv) / 15)) * 100 if hv else 50
    hock_vals = [m.get("hockAngle", 0) for m in per_frame if m.get("hockAngle")]
    hke = min(100, (_rom(hock_vals) / (20 * sf)) * 100) if hock_vals else 50
    sh_vals = [m.get("shoulderAngle", 0) for m in per_frame if m.get("shoulderAngle")]
    sfs = min(100, (_rom(sh_vals) / (15 * sf)) * 100) if sh_vals else 50
    overall = (symmetry * 20 + rhythm * 20 + tls * 15 + hcs * 15 + hke * 15 + sfs * 15) / 100
    return {
        "overall": round(min(100, overall), 1),
        "metrics": {
            "symmetry": round(symmetry, 1), "rhythm": round(rhythm, 1),
            "toplineStability": round(tls, 1), "headCarriage": round(hcs, 1),
            "hockEngagement": round(hke, 1), "shoulderFreedom": round(sfs, 1),
        },
        "foreDeltas": [round(d, 4) for d in fore_d[:300]],
        "hindDeltas": [round(d, 4) for d in hind_d[:300]],
        "strides": max(1, len(fore_d) // 8),
    }


def compute_bai(conf_score, move_score):
    CBS = max(0, min(100, conf_score))
    PAS = max(0, min(100, move_score))
    nicking = 8 if (CBS > 70 and PAS > 60) else (4 if CBS > 60 else 0)
    PIS = max(0, min(100, CBS * 0.4 + PAS * 0.3 + 30 + nicking))
    MVS = max(0, min(100, PIS * 0.3 + CBS * 0.3 + PAS * 0.2 + 20))
    bai = PIS * 0.35 + PAS * 0.30 + CBS * 0.20 + MVS * 0.15
    if bai >= 90: band = "EXCEPTIONAL"
    elif bai >= 80: band = "OUTSTANDING"
    elif bai >= 70: band = "VERY STRONG"
    elif bai >= 60: band = "ABOVE AVG"
    elif bai >= 50: band = "AVERAGE"
    else: band = "BELOW AVG"
    return {"bai": round(bai, 1), "band": band,
            "PIS": round(PIS), "PAS": round(PAS), "CBS": round(CBS), "MVS": round(MVS)}


def detect_risks(m):
    flags = []
    sa = m.get("shoulderAngle")
    if sa is not None and (sa < 40 or sa > 65):
        flags.append({"area": "Shoulder", "level": "high", "note": f"Angle {sa:.1f}\u00B0 outside 50-55\u00B0 optimal"})
    ha = m.get("hockAngle")
    if ha is not None and (ha < 140 or ha > 170):
        flags.append({"area": "Hock", "level": "high", "note": f"Geometry ({ha:.1f}\u00B0) distal limb stress"})
    fk = m.get("foreKneeAngle")
    if fk is not None and fk < 155:
        flags.append({"area": "Fore Knee", "level": "mod", "note": f"Perpendicularity ({fk:.1f}\u00B0)"})
    td = m.get("toplineDeviation")
    if td is not None and abs(td) > 5:
        flags.append({"area": "Topline", "level": "mod", "note": f"Deviation {td:.1f}"})
    return flags


# ═══════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="BloodstockAI", version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("BAI_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API info + health ──

@app.get("/api")
def api_info():
    return {
        "service": "BloodstockAI",
        "version": "3.1.0",
        "dlc_available": False,
        "dlc_model": f"{DLC_MODEL}/{DLC_BACKBONE}",
        "runtime": "vercel-serverless",
        "note": "DLC processing requires external GPU worker",
    }


@app.get("/health")
def health():
    try:
        ensure_schema()
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM analyses")
        count = cur.fetchone()["cnt"]
        cur.close()
        conn.close()
        return {"status": "ok", "database": "connected", "total_analyses": count,
                "dlc_model": f"{DLC_MODEL}/{DLC_BACKBONE}"}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


# ── Upload + create job ──

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    gait: str = Form("trot"),
):
    ensure_schema()

    if gait not in VALID_GAITS:
        raise HTTPException(400, f"Invalid gait '{gait}'. Must be one of: {VALID_GAITS}")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXT}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large ({len(content) // 1024 // 1024}MB). Max: {MAX_FILE_SIZE // 1024 // 1024}MB")

    jid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO analyses (id, created_at, updated_at, status, progress, filename, file_size, gait, dlc_model)
           VALUES (%s, %s, %s, 'pending_worker', 0, %s, %s, %s, %s)""",
        (jid, now, now, file.filename, len(content), gait, f"{DLC_MODEL}/{DLC_BACKBONE}"))
    conn.commit()
    cur.close()
    conn.close()

    # Write video to /tmp so worker webhook can retrieve it (if configured)
    tmp_path = Path(f"/tmp/{jid}{ext}")
    tmp_path.write_bytes(content)
    logger.info(f"[{jid}] Job created: {file.filename} ({len(content)} bytes) gait={gait}")

    # If worker URL is configured, notify it
    worker_url = os.environ.get("BAI_WORKER_URL")
    if worker_url:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{worker_url}/jobs/notify",
                data=json.dumps({"jobId": jid, "gait": gait}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning(f"[{jid}] Worker notification failed: {e}")

    return {"jobId": jid, "status": "pending_worker"}


# ── Status polling ──

@app.get("/status/{jid}")
def get_status(jid: str):
    ensure_schema()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT status, progress, error FROM analyses WHERE id = %s", (jid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(404, "Job not found")
    return {"jobId": jid, "status": row["status"], "progress": row["progress"], "error": row["error"]}


# ── Results ──

@app.get("/results/{jid}")
def get_results(jid: str):
    ensure_schema()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT status, error, results_json FROM analyses WHERE id = %s", (jid,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(404, "Job not found")
    if row["status"] == "error":
        raise HTTPException(500, row["error"] or "Analysis failed")
    if row["status"] != "complete":
        return JSONResponse(status_code=202, content={"_pending": True, "status": row["status"]})
    if not row["results_json"]:
        raise HTTPException(500, "Results data missing")
    return json.loads(row["results_json"])


# ── History ──

@app.get("/history")
def list_analyses(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    ensure_schema()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, created_at, status, filename, gait, bai_score, bai_band,
                  pis, pas, cbs, mvs, conf_overall, movement_overall, risk_flags,
                  video_duration, video_fps, sampled_frames
           FROM analyses ORDER BY created_at DESC LIMIT %s OFFSET %s""",
        (limit, offset))
    rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) as cnt FROM analyses")
    count = cur.fetchone()["cnt"]
    cur.close()
    conn.close()

    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "createdAt": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
            "status": r["status"],
            "filename": r["filename"], "gait": r["gait"],
            "baiScore": r["bai_score"], "baiBand": r["bai_band"],
            "pis": r["pis"], "pas": r["pas"], "cbs": r["cbs"], "mvs": r["mvs"],
            "confOverall": r["conf_overall"], "movementOverall": r["movement_overall"],
            "riskFlags": json.loads(r["risk_flags"]) if r["risk_flags"] else [],
            "videoDuration": r["video_duration"], "videoFps": r["video_fps"],
            "sampledFrames": r["sampled_frames"],
        })
    return {"items": items, "total": count, "limit": limit, "offset": offset}


# ── Delete job ──

@app.delete("/jobs/{jid}")
def delete_job(jid: str):
    ensure_schema()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM analyses WHERE id = %s", (jid,))
    conn.commit()
    cur.close()
    conn.close()
    return {"deleted": jid}


# ── Worker callback: receives completed results from GPU worker ──

@app.post("/api/worker/complete")
async def worker_complete(request: Request):
    """Called by the GPU worker when DLC processing finishes."""
    body = await request.json()
    jid = body.get("jobId")
    if not jid:
        raise HTTPException(400, "Missing jobId")

    auth = request.headers.get("Authorization")
    expected = os.environ.get("BAI_WORKER_SECRET")
    if expected and auth != f"Bearer {expected}":
        raise HTTPException(403, "Invalid worker secret")

    ensure_schema()
    conn = get_db()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    if "error" in body and body["error"]:
        cur.execute(
            "UPDATE analyses SET status='error', error=%s, updated_at=%s WHERE id=%s",
            (body["error"], now, jid))
    else:
        bai = body.get("baiScore", {})
        cur.execute(
            """UPDATE analyses SET status='complete', progress=100,
               bai_score=%s, bai_band=%s, pis=%s, pas=%s, cbs=%s, mvs=%s,
               conf_overall=%s, movement_overall=%s, risk_flags=%s,
               results_json=%s, video_width=%s, video_height=%s,
               video_fps=%s, video_duration=%s, video_total_frames=%s,
               sampled_frames=%s, updated_at=%s
               WHERE id=%s""",
            (bai.get("bai"), bai.get("band"), bai.get("PIS"), bai.get("PAS"),
             bai.get("CBS"), bai.get("MVS"),
             body.get("confOverall"), body.get("movementOverall"),
             json.dumps(body.get("riskFlags", [])),
             json.dumps(body.get("results")),
             body.get("videoWidth"), body.get("videoHeight"),
             body.get("videoFps"), body.get("videoDuration"),
             body.get("videoTotalFrames"), body.get("sampledFrames"),
             now, jid))

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"[{jid}] Worker callback: {'error' if body.get('error') else 'complete'}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

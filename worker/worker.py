"""
BloodstockAI GPU Worker
Runs on a machine with NVIDIA GPU + DeepLabCut installed.
Polls the Postgres database for pending jobs, processes them, and writes results back.

Usage:
    export POSTGRES_URL="postgresql://user:pass@host/db"
    export BAI_VERCEL_URL="https://your-app.vercel.app"  # optional: callback
    export BAI_WORKER_SECRET="your-secret"                # optional: auth
    python worker.py
"""
import os
import sys
import json
import math
import time
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    import deeplabcut
    DLC_OK = True
except ImportError:
    DLC_OK = False
    print("ERROR: DeepLabCut not installed. Run: pip install 'deeplabcut[pytorch]>=3.0'")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bai-worker")

POLL_INTERVAL = int(os.environ.get("BAI_POLL_INTERVAL", "5"))
DLC_MODEL = os.environ.get("BAI_DLC_MODEL", "superanimal_quadruped")
DLC_BACKBONE = os.environ.get("BAI_DLC_BACKBONE", "hrnet_w32")
DLC_DETECTOR = os.environ.get("BAI_DLC_DETECTOR", "fasterrcnn_resnet50_fpn_v2")
DLC_PCUTOFF = float(os.environ.get("BAI_DLC_PCUTOFF", "0.15"))
DLC_VIDEO_ADAPT = os.environ.get("BAI_DLC_VIDEO_ADAPT", "false").lower() == "true"
MAX_FRAMES = int(os.environ.get("BAI_MAX_FRAMES", "450"))

VERCEL_URL = os.environ.get("BAI_VERCEL_URL", "")
WORKER_SECRET = os.environ.get("BAI_WORKER_SECRET", "")


def pg_url():
    url = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL") or ""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def get_db():
    return psycopg2.connect(pg_url(), cursor_factory=RealDictCursor)


# ── DLC keypoint mapping (must match api/index.py) ──
DLC_TO_BAI = {
    "nose": "nose", "upper_jaw": "nose", "right_eye": "eye", "left_eye": "eye",
    "right_eartip": "nearear", "left_eartip": "farear",
    "right_earbase": "nearear", "left_earbase": "farear",
    "throat": "throat", "chin": "throat",
    "withers": "withers", "spine_mid": "back", "tailbase": "tailbase",
    "right_front_elbow": "elbow", "right_front_knee": "foreknee",
    "right_front_fetlock": "forefetlock", "right_front_hoof": "forehoof",
    "right_back_knee": "stifle",
    "right_back_fetlock": "hindfetlock", "right_back_hoof": "hindhoof",
}
DERIVED = {
    "poll": ("right_earbase", "left_earbase"),
    "shoulder": ("withers", "right_front_elbow"),
    "loin": ("spine_mid", "tailbase"),
    "croup": ("spine_mid", "tailbase", 0.67),
    "hip": ("tailbase", "right_back_elbow"),
    "hock": ("right_back_knee", "right_back_fetlock"),
    "girth": ("right_front_elbow", "right_back_elbow"),
}
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


def map_dlc_to_bai(dlc_kps):
    bai = {}
    for dn, bn in DLC_TO_BAI.items():
        if dn in dlc_kps and bn not in bai:
            x, y, c = dlc_kps[dn]
            if c > DLC_PCUTOFF:
                bai[bn] = (x, y, c)
    for bn, src in DERIVED.items():
        if bn in bai:
            continue
        if len(src) == 2:
            a, b = src
            if a in dlc_kps and b in dlc_kps:
                ax, ay, ac = dlc_kps[a]
                bx, by, bc = dlc_kps[b]
                if ac > DLC_PCUTOFF and bc > DLC_PCUTOFF:
                    bai[bn] = ((ax + bx) / 2, (ay + by) / 2, min(ac, bc))
        elif len(src) == 3:
            a, b, r = src
            if a in dlc_kps and b in dlc_kps:
                ax, ay, ac = dlc_kps[a]
                bx, by, bc = dlc_kps[b]
                if ac > DLC_PCUTOFF and bc > DLC_PCUTOFF:
                    bai[bn] = (ax + (bx - ax) * r, ay + (by - ay) * r, min(ac, bc))
    result = []
    for kp in BAI_KPS:
        if kp in bai:
            x, y, c = bai[kp]
            result.append({"id": kp, "label": BAI_LABELS.get(kp, kp),
                           "x": float(x), "y": float(y), "conf": float(c), "g": BAI_GRP[kp]})
        else:
            result.append({"id": kp, "label": BAI_LABELS.get(kp, kp),
                           "x": 0, "y": 0, "conf": 0, "g": BAI_GRP[kp]})
    return result


# ── Scoring (duplicated from api/index.py for self-contained worker) ──
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
    has = lambda *ids: all(i in lk for i in ids)
    if has("nose", "poll", "withers"): m["headNeckAngle"] = _ang(lk["nose"], lk["poll"], lk["withers"])
    if has("withers", "shoulder", "elbow"): m["shoulderAngle"] = _ang(lk["withers"], lk["shoulder"], lk["elbow"])
    if has("croup", "hip", "stifle"): m["hipAngle"] = _ang(lk["croup"], lk["hip"], lk["stifle"])
    if has("stifle", "hock", "hindfetlock"): m["hockAngle"] = _ang(lk["stifle"], lk["hock"], lk["hindfetlock"])
    if has("elbow", "foreknee", "forefetlock"): m["foreKneeAngle"] = _ang(lk["elbow"], lk["foreknee"], lk["forefetlock"])
    if has("withers", "back", "loin", "croup"):
        m["toplineDeviation"] = ((lk["back"][1] + lk["loin"][1]) / 2 - (lk["withers"][1] + lk["croup"][1]) / 2) * 100
    if has("shoulder", "hip", "withers", "forehoof"):
        d = _dst(lk["withers"], lk["forehoof"])
        m["bodyLengthRatio"] = _dst(lk["shoulder"], lk["hip"]) / d if d > 0 else 0
    if has("poll", "withers", "forehoof"):
        d = _dst(lk["withers"], lk["forehoof"])
        m["neckLengthRatio"] = _dst(lk["poll"], lk["withers"]) / d if d > 0 else 0
    if has("croup", "tailbase"):
        m["croupAngle"] = abs(math.degrees(math.atan2(lk["tailbase"][1] - lk["croup"][1], lk["tailbase"][0] - lk["croup"][0])))
    return m

SCORE_DEFS = [
    ("shoulderAngle", 95, 30, 1.5), ("headNeckAngle", 110, 40, 1.0),
    ("hipAngle", 95, 30, 1.5), ("hockAngle", 155, 30, 1.2),
    ("foreKneeAngle", 170, 25, 1.2), ("toplineDeviation", 0, 8, 1.3),
    ("bodyLengthRatio", 1.0, 0.35, 1.0), ("neckLengthRatio", 0.38, 0.2, 0.8),
    ("croupAngle", 25, 20, 1.0),
]
SCORE_LABELS = {
    "shoulderAngle": ("Shoulder Angle", "\u224895\u00B0"), "headNeckAngle": ("Head-Neck", "\u2248110\u00B0"),
    "hipAngle": ("Hip Angle", "\u224895\u00B0"), "hockAngle": ("Hock Angle", "\u2248155\u00B0"),
    "foreKneeAngle": ("Fore Knee", "\u2248170\u00B0"), "toplineDeviation": ("Topline Dev.", "\u22480"),
    "bodyLengthRatio": ("Body Ratio", "\u22481.0"), "neckLengthRatio": ("Neck Ratio", "\u22480.38"),
    "croupAngle": ("Croup Angle", "\u224825\u00B0"),
}

def score_conformation(metrics):
    scores = {}
    ts = tm = 0
    for key, ideal, tol, weight in SCORE_DEFS:
        if key in metrics:
            raw = max(0, 1 - abs(metrics[key] - ideal) / tol)
            sc = raw * 10 * weight
            mx = 10 * weight
            label, ideal_str = SCORE_LABELS.get(key, (key, ""))
            scores[key] = {"score": round(sc, 2), "maxScore": round(mx, 2), "raw": round(raw, 3),
                           "value": round(metrics[key], 2), "ideal": ideal_str, "label": label,
                           "pct": round((sc / mx) * 100, 1) if mx > 0 else 0}
            ts += sc
            tm += mx
    return scores, round(ts / tm * 100, 1) if tm > 0 else 0

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
    sym = (min(af, ah) / max(af, ah) * 100) if af > 0 and ah > 0 else 0
    all_d = fore_d + hind_d
    rhy = max(0, (1 - _std(all_d) / (_avg(all_d) or 1))) * 100
    per_frame = [compute_conformation(kps) for kps in frame_kps]
    tls = max(0, (1 - _std([m.get("toplineDeviation", 0) for m in per_frame]) / 3)) * 100
    hv = [m.get("headNeckAngle", 0) for m in per_frame if m.get("headNeckAngle")]
    hcs = max(0, (1 - _std(hv) / 15)) * 100 if hv else 50
    hk = [m.get("hockAngle", 0) for m in per_frame if m.get("hockAngle")]
    hke = min(100, (_rom(hk) / (20 * sf)) * 100) if hk else 50
    sv = [m.get("shoulderAngle", 0) for m in per_frame if m.get("shoulderAngle")]
    sfs = min(100, (_rom(sv) / (15 * sf)) * 100) if sv else 50
    ov = (sym * 20 + rhy * 20 + tls * 15 + hcs * 15 + hke * 15 + sfs * 15) / 100
    return {
        "overall": round(min(100, ov), 1),
        "metrics": {"symmetry": round(sym, 1), "rhythm": round(rhy, 1),
                    "toplineStability": round(tls, 1), "headCarriage": round(hcs, 1),
                    "hockEngagement": round(hke, 1), "shoulderFreedom": round(sfs, 1)},
        "foreDeltas": [round(d, 4) for d in fore_d[:300]],
        "hindDeltas": [round(d, 4) for d in hind_d[:300]],
        "strides": max(1, len(fore_d) // 8),
    }

def compute_bai(cs, ms):
    CBS = max(0, min(100, cs))
    PAS = max(0, min(100, ms))
    n = 8 if (CBS > 70 and PAS > 60) else (4 if CBS > 60 else 0)
    PIS = max(0, min(100, CBS * 0.4 + PAS * 0.3 + 30 + n))
    MVS = max(0, min(100, PIS * 0.3 + CBS * 0.3 + PAS * 0.2 + 20))
    bai = PIS * 0.35 + PAS * 0.30 + CBS * 0.20 + MVS * 0.15
    if bai >= 90: band = "EXCEPTIONAL"
    elif bai >= 80: band = "OUTSTANDING"
    elif bai >= 70: band = "VERY STRONG"
    elif bai >= 60: band = "ABOVE AVG"
    elif bai >= 50: band = "AVERAGE"
    else: band = "BELOW AVG"
    return {"bai": round(bai, 1), "band": band, "PIS": round(PIS), "PAS": round(PAS), "CBS": round(CBS), "MVS": round(MVS)}

def detect_risks(m):
    f = []
    if m.get("shoulderAngle") and (m["shoulderAngle"] < 40 or m["shoulderAngle"] > 65):
        f.append({"area": "Shoulder", "level": "high", "note": f"Angle {m['shoulderAngle']:.1f}\u00B0 outside optimal"})
    if m.get("hockAngle") and (m["hockAngle"] < 140 or m["hockAngle"] > 170):
        f.append({"area": "Hock", "level": "high", "note": f"Geometry ({m['hockAngle']:.1f}\u00B0) distal limb stress"})
    if m.get("foreKneeAngle") and m["foreKneeAngle"] < 155:
        f.append({"area": "Fore Knee", "level": "mod", "note": f"Perpendicularity ({m['foreKneeAngle']:.1f}\u00B0)"})
    if m.get("toplineDeviation") and abs(m["toplineDeviation"]) > 5:
        f.append({"area": "Topline", "level": "mod", "note": f"Deviation {m['toplineDeviation']:.1f}"})
    return f


# ── Main processing loop ──
def process_job(job):
    jid = job["id"]
    gait = job["gait"] or "trot"
    logger.info(f"[{jid}] Processing {job['filename']} gait={gait}")

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("UPDATE analyses SET status='running_dlc', progress=10 WHERE id=%s", (jid,))
        conn.commit()

        # Video must be accessible — check /tmp (if Vercel stored it) or download from URL
        # For now, worker needs the video file to exist locally.
        # You can extend this to download from S3/Blob/etc.
        video_path = None
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            p = Path(f"/tmp/{jid}{ext}")
            if p.exists():
                video_path = p
                break

        if not video_path:
            raise FileNotFoundError(f"Video file not found for job {jid}. "
                                    "Ensure video is accessible at /tmp/{jid}.<ext> "
                                    "or configure shared storage.")

        # Read video metadata
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()
        step = max(1, total_frames // MAX_FRAMES)

        cur.execute(
            "UPDATE analyses SET video_width=%s, video_height=%s, video_fps=%s, "
            "video_duration=%s, video_total_frames=%s, sampled_frames=%s, progress=15 WHERE id=%s",
            (w, h, fps, round(duration, 2), total_frames, min(total_frames, MAX_FRAMES), jid))
        conn.commit()

        # Run DLC
        dlc_out = Path(tempfile.mkdtemp(prefix=f"bai_{jid}_"))
        try:
            deeplabcut.video_inference_superanimal(
                [str(video_path)], DLC_MODEL,
                model_name=DLC_BACKBONE, detector_name=DLC_DETECTOR,
                videotype=video_path.suffix, video_adapt=DLC_VIDEO_ADAPT,
                scale_list=range(200, 600, 50), destfolder=str(dlc_out))
        except TypeError:
            deeplabcut.video_inference_superanimal(
                [str(video_path)], DLC_MODEL,
                model_name=DLC_BACKBONE, detector_name=DLC_DETECTOR,
                videotype=video_path.suffix, video_adapt=DLC_VIDEO_ADAPT,
                scale_list=range(200, 600, 50), dest_folder=str(dlc_out))

        cur.execute("UPDATE analyses SET status='parsing_keypoints', progress=60 WHERE id=%s", (jid,))
        conn.commit()

        # Parse H5
        h5s = list(dlc_out.glob("*.h5")) or list(video_path.parent.glob(f"*{video_path.stem}*.h5"))
        if not h5s:
            raise FileNotFoundError("DLC produced no output .h5 file")
        df = pd.read_hdf(h5s[0])
        scorer = df.columns.get_level_values(0)[0]

        all_frames = []
        idxs = list(range(0, min(total_frames, len(df)), step))
        for ii, fi in enumerate(idxs):
            if fi >= len(df):
                break
            row = df.iloc[fi]
            dlc_kps = {}
            for bp in df.columns.get_level_values(1).unique():
                try:
                    x = float(row[(scorer, bp, "x")])
                    y = float(row[(scorer, bp, "y")])
                    lk = float(row[(scorer, bp, "likelihood")])
                    dlc_kps[bp] = (x / w, y / h, lk)
                except:
                    continue
            all_frames.append({"fi": fi, "t": round(fi / fps, 3), "kps": map_dlc_to_bai(dlc_kps)})

            if ii % 50 == 0:
                p = 60 + int((ii / max(1, len(idxs))) * 25)
                cur.execute("UPDATE analyses SET progress=%s WHERE id=%s", (p, jid))
                conn.commit()

        if not all_frames:
            raise ValueError("No valid keypoints extracted")

        cur.execute("UPDATE analyses SET status='computing_scores', progress=88 WHERE id=%s", (jid,))
        conn.commit()

        # Score
        mid_kps = all_frames[len(all_frames) // 2]["kps"]
        conf_m = compute_conformation(mid_kps)
        conf_scores, conf_ov = score_conformation(conf_m)
        movement = compute_movement([f["kps"] for f in all_frames], gait)
        bai = compute_bai(conf_ov, movement["overall"])
        risks = detect_risks(conf_m)

        result = {
            "jobId": jid,
            "videoInfo": {"width": w, "height": h, "fps": round(fps, 2),
                          "duration": round(duration, 2), "totalFrames": total_frames,
                          "sampledFrames": len(all_frames)},
            "dlcModel": f"{DLC_MODEL}/{DLC_BACKBONE}",
            "frames": all_frames, "frameCount": len(all_frames), "fps": round(fps, 2),
            "gait": gait, "baiScore": bai,
            "conformation": {"metrics": conf_m, "scores": conf_scores, "overall": conf_ov},
            "movement": movement, "riskFlags": risks,
        }

        results_json = json.dumps(result, default=str)
        cur.execute(
            """UPDATE analyses SET status='complete', progress=100,
               bai_score=%s, bai_band=%s, pis=%s, pas=%s, cbs=%s, mvs=%s,
               conf_overall=%s, movement_overall=%s, risk_flags=%s,
               results_json=%s, updated_at=NOW() WHERE id=%s""",
            (bai["bai"], bai["band"], bai["PIS"], bai["PAS"], bai["CBS"], bai["MVS"],
             conf_ov, movement["overall"], json.dumps(risks), results_json, jid))
        conn.commit()

        logger.info(f"[{jid}] Complete. BAI: {bai['bai']} ({bai['band']})")

    except Exception as e:
        logger.error(f"[{jid}] Failed: {e}", exc_info=True)
        cur.execute("UPDATE analyses SET status='error', error=%s, progress=0, updated_at=NOW() WHERE id=%s",
                    (str(e), jid))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def main():
    url = pg_url()
    if not url:
        print("ERROR: Set POSTGRES_URL environment variable")
        sys.exit(1)

    logger.info(f"BloodstockAI GPU Worker started")
    logger.info(f"  DLC: {DLC_MODEL}/{DLC_BACKBONE}")
    logger.info(f"  Polling every {POLL_INTERVAL}s")
    logger.info(f"  DB: {url[:30]}...")

    while True:
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM analyses WHERE status='pending_worker' ORDER BY created_at ASC LIMIT 1")
            job = cur.fetchone()
            cur.close()
            conn.close()

            if job:
                process_job(job)
            else:
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Poll error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

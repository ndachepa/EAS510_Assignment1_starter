"""EAS 510 - Project 1: Rule functions for the Digital Forensics Apprentice.

Each rule compares one suspect image against one registered original and
returns a dict:

    {
        "rule": 1,
        "name": "Metadata",
        "fired": bool,          # True if the rule found supporting evidence
        "score": int,           # points awarded out of "out_of"
        "out_of": int,
        "note": "Size ratio 0.85",   # short human-readable metric
        "metric": 0.85,              # raw similarity measure in [0, 1]
    }

The points budgets are fixed by the assignment: Rule 1 = 30, Rule 2 = 30,
Rule 3 = 40. The detector sums them into a 0-100 confidence score.

These starter implementations are deliberately WEAK baselines (they hinge on
simple thresholds). Improving them is the assignment.

Reads are cached per path so one suspect against many originals does not
re-decode images needlessly.
"""

import os
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image

#: Rule names used by SimpleDetector.evaluate() (V1).
RULES = ("rule1_metadata", "rule2_histogram", "rule3_template")


@lru_cache(maxsize=256)
def _arr(path):
# Some originals are 7000+ pixels wide, and loading them at full size was
# slow and ate a lot of memory. We never compare at more than ~1024 px
# anyway, so I load big files at a smaller size right from the start.
    size = _size(path)
    if size is None:
        return None
    if max(size) >= 4096:
        # really big photo: let OpenCV decode it at 1/4 of the size
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    elif max(size) >= 2048:
        # medium photo: decode at half size
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    else:
        # small enough already, load it normally
        img = cv2.imread(path)
    if img is None:
        return None
    # final trim so the longest side is at most 1024 px
    return _downscale(img, 1024)

@lru_cache(maxsize=256)
def _hist(path):
    img = _arr(path)
    if img is None:
        return None
    hists = [cv2.calcHist([img], [i], None, [32], [0, 256]) for i in range(3)]
    hist_all = np.concatenate(hists).ravel().astype(np.float32)
    cv2.normalize(hist_all, hist_all)
    return hist_all


def _downscale(img, max_dim=512):
    """Fit an image into `max_dim` before expensive cv2 work."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
# INTER_AREA averages neighbouring pixels while shrinking, so the
# small version stays smooth instead of jagged (helps the matching)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


@lru_cache(maxsize=256)
def _gray(path):
    img = _arr(path)
    if img is None:
        return None
    return _downscale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


@lru_cache(maxsize=256)
def _size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None

def _is_png(path):
    # Evert file announce its type in its first few bytes:
    # PNG starts with 89 50 4E 47 and JPEG starts with FF D8.
    # I need this because the PNG copies have the exacr same pixels as the orginals.
    # But there are -7x bigger on disk, so file size alone would make worst for Rule 1.
    with open(path, "rb") as f:
      return f.read(4) == b"\x89PNG"

## RULE-1. Metadata
def rule1_metadata(target, input_path):
    """File size + dimensions. Compression and crops leave size fingerprints."""
    out = {"rule": 1, "name": "Metadata", "fired": False, "score": 0,
           "out_of": 30, "note": "Size ratio 0.00", "metric": 0.0}
    try:
        src_w, src_h = _size(target["path"]) or (0,0)
        in_w, in_h = _size(input_path) or (0,0)
        # 1) Dimensions: same size, or smaller with the same shape (crop)?
        if (in_w, in_h) == (src_w, src_h):
            dims = 1.0
        elif in_w <= src_w and in_h <= src_h and abs(in_w / src_w - in_h / src_h) < 0.01:
            dims = 0.7
        else:
            dims = 0.0
        # 2) File size per pixels - omly fair to compare if both are the same format
        if _is_png(target["path"]) != _is_png(input_path):
            size = 1.0 if dims == 1.0 else 0.5
        else:
            src_bpp = os.path.getsize(target["path"]) / (src_w * src_h)
            in_bpp = os.path.getsize(input_path) / (in_w * in_h)
            size = min(src_bpp, in_bpp) / max(src_bpp, in_bpp)
        metric = 0.7 * dims + 0.3 * size
        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        out["note"] = f"Size ratio {out['metric']:.2f}"
        if out["metric"] >= 0.6:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out

## RULE-2 Histogram 
def rule2_histogram(target, input_path):
    """Color histogram correlation. Robust to crops and spatial changes."""
    out = {"rule": 2, "name": "Histogram", "fired": False, "score": 0,
           "out_of": 30, "note": "Correlation 0.00", "metric": 0.0}
    try:
        hs, hi = _hist(target["path"]), _hist(input_path)
        if hs is None or hi is None:
            return out
        corr = float(cv2.compareHist(hs, hi, cv2.HISTCMP_CORREL))
        out["metric"] = round(max(0.0, min(1.0, corr)), 3)
        out["note"] = f"Correlation {out['metric']:.2f}"
# I raised this from 0.5 to 0.7: random photos never got above ~0.2,
# while real copies usually land between 0.8 and 1.0.
        if out["metric"] >= 0.7:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out

## RULE-3 Template
def rule3_template(target, input_path):
    """Template matching. Detects when the suspect is contained in the original.

    cv2.matchTemplate needs the template (suspect) smaller than the target
    (original); on rotated/resized suspects this rule degrades fast. Fixing
    that becomes Rule 4 territory.
    """
    out = {"rule": 3, "name": "Template", "fired": False, "score": 0,
           "out_of": 40, "note": "Match score 0.00", "metric": 0.0}
    try:
        src_g = _gray(target["path"])
        nd_g = _gray(input_path)
        if src_g is None or nd_g is None:
            return out
        # This was the main bug. Both images used to be shrunk to 512 px on their own
        # So a 25% crop got blown up to the same size as the whole
        # original - like looking for a puzzle piece that's 4x too big.
        # Fix: work out how much the ORIGINAL was shrunk..
        scale = src_g.shape[1] / _size(target["path"])[0]
        # ..and shrink the suspect by that same amount, so the crop is the 
        # same size as the matching area inside the orginal.
        in_w, in_h = _size(input_path)
        nd_g = cv2.resize(nd_g, (max(1, round(in_w * scale)), max(1, round(in_h * scale))),
                          interpolation=cv2.INTER_AREA)
        if (src_g.shape[0] < nd_g.shape[0]) or (src_g.shape[1] < nd_g.shape[1]):
            nd_g = cv2.resize(nd_g, (min(nd_g.shape[1], src_g.shape[1]),
                                     min(nd_g.shape[0], src_g.shape[0])))
        res = cv2.matchTemplate(src_g, nd_g, cv2.TM_CCOEFF_NORMED)
        metric = float(res.max())
        out["metric"] = round(max(0.0, min(1.0, metric)), 3)
        out["note"] = f"Match score {out['metric']:.2f}"
        # real matches score 0.97-1.00 and the best wrong orginal only got 0.73
        # so, 0.8 sits perfectly between (was 0.4)
        if out["metric"] >= 0.8:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * out["metric"]))
    except Exception:
        pass
    return out
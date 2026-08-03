#!/usr/bin/env python3
"""Find the fruit in a rendered frame, and turn a box into a 3D position.

Step 3 proved the geometry against pixels the simulator handed over for free.
This is where a detector goes in front of it, and from here every millimetre of
error that appears is the detector's — which is the entire reason the ordering
is geometry first.

⚠️ **Name the domain gap before reading any number out of this file.** MuJoCo's
renderer is basic; that was a known, accepted cost of dropping Gazebo and ruling
out Isaac Sim, and it is written into the settled decisions as *no photoreal
CV-in-sim, the detector trains on real imagery*. A tomato in this scene is a
uniform red sphere with one specular highlight, under three flat lights, with no
texture, no calyx, no ribbing, no ripeness variation and no bruises. So:

  * a detector trained on **real** tomatoes may not fire on these spheres at
    all, because they do not look like tomatoes;
  * a detector trained on **these spheres** would hit ~100% and prove nothing
    whatsoever about a greenhouse.

Both are true at once and neither is a bug to fix. What the detector is here to
prove is **the pipeline, not the field accuracy**. Week 3's milestone is a 3D
coordinate that matches ground truth; detection quality on real fruit is a
Week-5-and-beyond problem with a real dataset behind it.

Two detectors, and keeping both is the point:

    A. `HSVDetector`  — colour threshold and contour fit, OpenCV, no model.
       The **control**. It never fails on this scene, which is simultaneously
       its weakness and its whole value: when it is the front end, any error
       downstream is geometric, because the detector contributed none.
    B. `YoloDetector`  — pretrained yolo11n, COCO classes. The honest attempt.
       Recall on rendered spheres is unpredictable and is allowed to be bad;
       what it proves is that a real detector's output format feeds the rest of
       the pipeline unchanged.

Scored on **recall and false positives per frame**, never mAP. A harvester has
two failure modes with very different costs: a missed fruit is lost revenue and
nothing worse — it gets picked next pass or never — while a false positive
sends the arm into the canopy at a fruit that is not there, and Week 2 spent
the entire week making the arm not do that. Averaging those two into one number
throws away the only distinction that matters.

    ./.venv/bin/python simulation/mujoco/detect.py              # both, scored
    ./.venv/bin/python simulation/mujoco/detect.py --save /tmp/d
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camera import DEPTH_MAX, deproject, project  # noqa: E402
from plant_row import FRUIT_R  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "weights" / "yolo11n.pt"

# --- option A thresholds -----------------------------------------------------
# Sampled off the rendered fruit at three poses: H = 2, S = 177-198, V = 144-206
# on a 0-179 / 0-255 / 0-255 OpenCV scale. Red straddles the hue wrap, so it
# takes two bands — the classic way to lose half your detections is to write
# one.
#
# The bands are deliberately loose. Tight ones would score better here and
# would be tuning to a renderer, which is the one thing this detector must not
# do: it exists to be a control, and a control that has been fitted to the test
# set is not a control.
HSV_LO_1, HSV_HI_1 = (0, 110, 60), (10, 255, 255)
HSV_LO_2, HSV_HI_2 = (170, 110, 60), (179, 255, 255)

MIN_AREA_PX = 200        # ~8 px radius; smaller than the arm can act on anyway
MIN_CIRCULARITY = 0.55   # 4*pi*A/P^2; a sphere is 1.0, a leaf edge is far less

# A detection counts as matching a ground-truth fruit if their centres are
# within this many *ground-truth radii*. Centre distance rather than IoU on
# purpose: what the pipeline consumes is the centre pixel, so the thing worth
# scoring is whether the centre is right, not whether two rectangles agree.
MATCH_RADII = 1.0


@dataclass
class Detection:
    """One box, in pixels, plus whatever the detector wants to say about it."""

    u0: float
    v0: float
    u1: float
    v1: float
    score: float = 1.0
    label: str = "fruit"
    source: str = ""
    # Filled by `estimate`; kept on the detection so the association step in
    # week3_perceive has one object to carry around.
    est: np.ndarray = None
    est_r: float = float("nan")
    depth_m: float = float("nan")
    matched: str = None
    extra: dict = field(default_factory=dict)

    @property
    def centre(self):
        return np.array([(self.u0 + self.u1) / 2, (self.v0 + self.v1) / 2])

    @property
    def width(self):
        return self.u1 - self.u0

    @property
    def height(self):
        return self.v1 - self.v0

    @property
    def radius_px(self):
        """Half the *smaller* side.

        The smaller one, because a box clipped by the frame edge or fused with a
        neighbouring blob stretches in one axis only, and a sphere's true
        projection is round. Taking the larger side would inflate the fitted
        radius exactly when the detection is worst.
        """
        return min(self.width, self.height) / 2


# --- A: the control ----------------------------------------------------------
class HSVDetector:
    """Red blobs that are round enough. No model, no weights, no training.

    Keep this forever, even after a real detector works. It is the instrument
    that answers "is this error the geometry or the detector?", and that
    question comes up every week from here on.
    """

    name = "hsv"

    def __init__(self, min_area=MIN_AREA_PX, min_circularity=MIN_CIRCULARITY):
        self.min_area = min_area
        self.min_circularity = min_circularity

    def __call__(self, rgb):
        import cv2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(HSV_LO_1), np.array(HSV_HI_1)),
            cv2.inRange(hsv, np.array(HSV_LO_2), np.array(HSV_HI_2)))
        # Open then close: opening drops the speckle the specular highlight
        # leaves around the rim, closing fills the highlight itself, which on a
        # smooth sphere under a point light is bright enough to fall out of the
        # saturation band and punch a hole in the middle of the fruit.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

        out = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
            if circ < self.min_circularity:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection(x, y, x + w, y + h, score=float(circ),
                                 source=self.name,
                                 extra={"area": float(area),
                                        "circularity": float(circ)}))
        return out


# --- B: the honest attempt ---------------------------------------------------
# COCO has no tomato. These are the classes a red sphere plausibly trips, and
# accepting all three is deliberate: the question is not "does YOLO know what a
# tomato is" — it cannot — but "does a real detector's output feed this
# pipeline". Restricting to one class would hide most of what it does.
YOLO_CLASSES = {"apple", "orange", "sports ball"}


class YoloDetector:
    """Pretrained yolo11n, COCO weights, no fine-tuning.

    ⚠️ Its recall here is not a statement about YOLO and not a statement about
    greenhouse perception. It is a statement about how far a COCO-trained model
    generalises to an untextured sphere under flat lighting, which is a question
    nobody needed answering. It is in the pipeline to prove the seam.

    Fine-tuning on rendered frames (option C in the instructions) would push
    this to whatever number is wanted and would mean nothing outside this
    renderer. It is only worth doing as a stepping stone to fine-tuning on real
    imagery, and it is not needed: option A already unblocks Steps 5 and 6.
    """

    name = "yolo"

    def __init__(self, weights=WEIGHTS, conf=0.10, classes=YOLO_CLASSES):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.classes = classes

    def __call__(self, rgb):
        # conf is set low on purpose. The interesting failure is "fires on a
        # heating pipe", and a high threshold hides it — which would make the
        # false-positive count look good for the wrong reason.
        res = self.model.predict(rgb, conf=self.conf, verbose=False)[0]
        names = res.names
        out = []
        for b in res.boxes:
            label = names[int(b.cls)]
            if self.classes and label not in self.classes:
                continue
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
            out.append(Detection(x0, y0, x1, y1, score=float(b.conf),
                                 label=label, source=self.name))
        # ⚠️ Class-agnostic NMS, and it is not cosmetic. YOLO suppresses
        # overlapping boxes *within* a class, so accepting three classes means
        # one tomato comes back as both "apple" and "sports ball" — measured,
        # that was most of this detector's false positives, and every one of
        # them was a duplicate on a real fruit rather than a phantom. Left in,
        # the FP count says the detector is dangerous when it is merely
        # indecisive about what a tomato is called.
        return nms(out)


def nms(dets, iou_thresh=0.55):
    """Class-agnostic non-maximum suppression, highest score wins."""
    keep = []
    for d in sorted(dets, key=lambda x: -x.score):
        if all(_iou(d, k) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def _iou(a, b):
    iw = max(0.0, min(a.u1, b.u1) - max(a.u0, b.u0))
    ih = max(0.0, min(a.v1, b.v1) - max(a.v0, b.v0))
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


# --- box -> metres -----------------------------------------------------------
def estimate(det, depth, intr, R, C, percentile=15.0):
    """Turn one box into a world position. Fills the detection in place.

    Two departures from Step 3's calibration path, and both exist because Step 3
    was allowed simulator luxuries that a detector is not:

    **Depth from a percentile over the box, not from the centre pixel.** The
    centre pixel is one sample. A leaf crossing the middle of the fruit, or a
    box whose centre has drifted off the blob, replaces the fruit's depth with
    something else's and the estimate is confidently wrong with nothing to flag
    it. A low percentile over the box's valid pixels asks "what is the nearest
    real surface in here", which is what the front of a tomato is. 15th rather
    than 0th because the minimum is a single pixel and single pixels are where
    render noise lives.

    **Radius fitted from the box width, not the known FRUIT_R.** The
    surface-to-centre correction needs a radius, and Step 3 took it from the
    constant — correct there, because Step 3 was testing geometry against a
    known sphere. A detector does not know the radius. Fitting it from the
    apparent width, r = (w_px / 2) * range / f, is the honest version and it
    degrades the way the real thing does: a clipped or fused box gives a wrong
    radius and the position error grows by exactly that amount. The alternative
    — assume every tomato is 33 mm — would score better here and would be
    encoding the answer.

    Returns the detection, with `est` None if the box has no usable depth.
    """
    u0 = max(0, int(np.floor(det.u0)))
    v0 = max(0, int(np.floor(det.v0)))
    u1 = min(intr.width, int(np.ceil(det.u1)))
    v1 = min(intr.height, int(np.ceil(det.v1)))
    if u1 <= u0 or v1 <= v0:
        return det

    # A box touching the border is a fruit the sensor only has part of. The
    # radius fit reads the *clipped* width, so the estimate is short by however
    # much is off-screen — and unlike a miss, it fails silently with a
    # plausible number. Flag it here; `match` reports it as its own bucket.
    det.extra["edge"] = (det.u0 <= 1 or det.v0 <= 1
                         or det.u1 >= intr.width - 1
                         or det.v1 >= intr.height - 1)

    patch = depth[v0:v1, u0:u1]
    valid = patch[patch < DEPTH_MAX]
    if valid.size == 0:
        # Every pixel in the box is the far plane — the detector boxed sky.
        # Returning None here is the mask that stops a fruit being reported
        # most of a kilometre outside the greenhouse.
        det.extra["reject"] = "all background"
        return det

    z = float(np.percentile(valid, percentile))
    u, v = det.centre

    # Fit the radius. `z` is z-depth; the range to the surface is what sets the
    # pixel scale, so convert before dividing.
    ray = np.array([(u - intr.cx) / intr.f, -(v - intr.cy) / intr.f, -1.0])
    ray /= np.linalg.norm(ray)
    rng = z / abs(ray[2])
    r_est = float(det.radius_px * rng / intr.f)

    # A sanity band, not a correction. Anything outside it is a box that is not
    # a tomato and the position it implies is meaningless; better to say so than
    # to hand the planner a number. The band is wide (8 mm to 90 mm) because
    # narrowing it to the known radius is the same cheat as assuming it.
    if not (0.008 <= r_est <= 0.090):
        det.extra["reject"] = f"fitted radius {r_est * 1000:.0f} mm"
        det.est_r = r_est
        det.depth_m = z
        return det

    det.depth_m = z
    det.est_r = r_est
    det.est = deproject(intr, R, C, u, v, z, radius=r_est)
    return det


# --- ground truth, free and exact -------------------------------------------
def truth_boxes(model, data, intr, R, C, names, occlusion=True):
    """Where each fruit *is*, as a box, straight out of the simulator.

    This is a genuinely good use of a simulator and it is what makes option C
    (fine-tuning on rendered frames) cheap if it is ever wanted: MuJoCo knows
    where every fruit is, so labelling a frame costs one projection. The boxes
    come out of Step 3's own maths, so the labels and the pipeline cannot drift
    apart.

    The projected outline of a sphere is an ellipse, not a circle, and the
    difference grows off-axis. At this working distance it is under a pixel, so
    the box is built as f*r/range square and the approximation is named rather
    than silently relied on.
    """
    from camera import occluder

    out = {}
    for n in names:
        p = data.body(n).xpos.copy()
        u, v = project(intr, R, C, p)
        rng = float(np.linalg.norm(p - C))
        r_px = intr.f * FRUIT_R / rng
        if not (0 <= u < intr.width and 0 <= v < intr.height):
            continue
        if occlusion and occluder(model, data, C, p, n) is not None:
            # Not in the denominator. Scoring a detector on a fruit the camera
            # physically cannot see measures the scene, not the detector.
            continue
        out[n] = {"uv": np.array([u, v]), "r_px": float(r_px),
                  "pos": p, "range": rng,
                  "box": (u - r_px, v - r_px, u + r_px, v + r_px)}
    return out


def match(dets, truth, radii=MATCH_RADII):
    """Greedy nearest-centre association between detections and known fruit.

    Greedy by ascending centre distance, each side used once. Good enough here
    because the fruit are 140 mm apart and never overlap in the image; the
    moment they cluster, this is the first thing that breaks and the honest fix
    is Hungarian assignment on a proper cost.

    Returns (recall, false_positives, per-detection assignment written in place).
    """
    pairs = []
    for i, d in enumerate(dets):
        for n, t in truth.items():
            dist = float(np.linalg.norm(d.centre - t["uv"]))
            if dist <= radii * t["r_px"]:
                pairs.append((dist, i, n))
    pairs.sort()

    used_d, used_t = set(), set()
    for _, i, n in pairs:
        if i in used_d or n in used_t:
            continue
        used_d.add(i)
        used_t.add(n)
        dets[i].matched = n

    # Classify what is left over. Lumping every unmatched box into one "false
    # positive" number hides the only distinction a harvester cares about:
    #
    #   duplicate — a second box on a fruit already found. Harmless: the
    #               association step in Step 6 collapses it, and the arm goes
    #               to the same place either way.
    #   edge      — a box touching the frame border, i.e. a fruit the camera
    #               has only half of. Its fitted radius is wrong by
    #               construction, so its position is not trustworthy — but
    #               there *is* a tomato there. The right response is to
    #               re-frame, not to drive at it.
    #   phantom   — nothing there. **This is the dangerous one**, and it is the
    #               number to quote: it is the box that sends the arm into the
    #               canopy at a fruit that does not exist.
    hits = len(used_t)
    buckets = {"duplicate": 0, "edge": 0, "phantom": 0}
    for i, d in enumerate(dets):
        if i in used_d:
            continue
        near = any(np.linalg.norm(d.centre - t["uv"]) <= 2.5 * t["r_px"]
                   for t in truth.values())
        if near:
            d.extra["fp"] = "duplicate"
        elif d.extra.get("edge"):
            d.extra["fp"] = "edge"
        else:
            d.extra["fp"] = "phantom"
        buckets[d.extra["fp"]] += 1
    return hits, buckets


# --- drawing -----------------------------------------------------------------
def draw(rgb, dets, truth=None, errors=None):
    """Overlay boxes, estimates and the ground-truth error, for the video.

    The frame worth having this week is not the arm — it is the detection with
    the 3D estimate and the error against truth printed on it. A grower does not
    read a success rate; they read "the robot can see the tomato and knows where
    it is, to eleven millimetres".
    """
    import cv2

    img = np.ascontiguousarray(rgb[:, :, ::-1])   # to BGR for cv2, and writable

    if truth:
        for n, t in truth.items():
            u, v = t["uv"]
            cv2.circle(img, (int(u), int(v)), 4, (0, 255, 255), -1)
            cv2.circle(img, (int(u), int(v)), int(t["r_px"]), (0, 190, 190), 1)

    for d in dets:
        ok = d.est is not None
        colour = (80, 220, 80) if ok else (60, 60, 220)
        cv2.rectangle(img, (int(d.u0), int(d.v0)), (int(d.u1), int(d.v1)),
                      colour, 2)
        tag = f"{d.source}:{d.label} {d.score:.2f}"
        if d.matched:
            tag = f"{d.matched} <- {tag}"
        cv2.putText(img, tag, (int(d.u0), max(14, int(d.v0) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if ok:
            line = (f"[{d.est[0]:.3f} {d.est[1]:.3f} {d.est[2]:.3f}] m"
                    f"  r={d.est_r * 1000:.0f}mm")
            cv2.putText(img, line, (int(d.u0), int(d.v1) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            if errors and d.matched in errors:
                e = errors[d.matched]
                cv2.putText(img, f"err {e * 1000:5.1f} mm",
                            (int(d.u0), int(d.v1) + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (60, 220, 250), 2, cv2.LINE_AA)
        elif "reject" in d.extra:
            cv2.putText(img, d.extra["reject"], (int(d.u0), int(d.v1) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return img


# --- the frame set -----------------------------------------------------------
def pose_set(model, park_q):
    """Arm poses to score over. Varied, and every one of them reachable.

    A recall figure from one pose is a recall figure for one pose. These sweep
    the staging plane across the row's whole y range and both ends of its
    height band, which is the set of poses the picker actually visits.
    """
    from mission import STAGE_X

    return [
        ("park", None),
        ("stage t0", np.array([STAGE_X, 0.22, 0.70])),
        ("stage t1", np.array([STAGE_X, 0.08, 0.62])),
        ("stage t2", np.array([STAGE_X, -0.08, 0.68])),
        ("stage t3", np.array([STAGE_X, -0.24, 0.62])),
        ("stage t4", np.array([STAGE_X, 0.26, 0.56])),
        ("stage high", np.array([STAGE_X - 0.06, 0.00, 0.76])),
        ("stage back", np.array([STAGE_X - 0.10, 0.10, 0.60])),
    ]


def score_detector(model, data, sensor, detector, park_q, row, poses=None,
                   save=None, verbose=True):
    """Run one detector over the pose set and report recall and FP/frame."""
    import mujoco

    from camera import stage

    poses = pose_set(model, park_q) if poses is None else poses
    intr = sensor.intr
    frames = 0
    gt_total = hits_total = 0
    totals = {"duplicate": 0, "edge": 0, "phantom": 0}
    errs = []

    if verbose:
        print(f"\n  detector '{detector.name}'")
        print(f"  {'pose':>11} {'in view':>8} {'found':>6} {'hit':>4} "
              f"{'dup':>4} {'edge':>5} {'phantom':>8} {'mean err mm':>12} "
              f"{'worst mm':>9}")

    for label, target in poses:
        from mission import reset_park
        if target is None:
            reset_park(model, data, park_q)
            row.reset()
            mujoco.mj_forward(model, data)
        else:
            stage(model, data, park_q, target, row)

        rgb, depth = sensor.both(data)
        R, C = sensor.pose(data)
        truth = truth_boxes(model, data, intr, R, C, row.names)
        dets = [estimate(d, depth, intr, R, C) for d in detector(rgb)]
        hits, fps = match(dets, truth)

        frames += 1
        gt_total += len(truth)
        hits_total += hits
        for k, v in fps.items():
            totals[k] += v

        errors = {}
        for d in dets:
            if d.matched and d.est is not None:
                e = float(np.linalg.norm(d.est - truth[d.matched]["pos"]))
                errors[d.matched] = e
                errs.append((d.matched, d.est - truth[d.matched]["pos"], e))

        if verbose:
            vals = list(errors.values())
            print(f"  {label:>11} {len(truth):8d} {len(dets):6d} {hits:4d} "
                  f"{fps['duplicate']:4d} {fps['edge']:5d} "
                  f"{fps['phantom']:8d} "
                  f"{(1000 * np.mean(vals) if vals else float('nan')):12.1f} "
                  f"{(1000 * max(vals) if vals else float('nan')):9.1f}")

        if save:
            import cv2
            Path(save).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save) / f"{detector.name}_"
                            f"{label.replace(' ', '_')}.png"),
                        draw(rgb, dets, truth, errors))

    recall = hits_total / gt_total if gt_total else float("nan")
    fp_total = sum(totals.values())
    return {
        "detector": detector.name, "frames": frames, "gt": gt_total,
        "hits": hits_total, "fp": fp_total, "recall": recall,
        "fp_per_frame": fp_total / frames if frames else float("nan"),
        "phantom_per_frame": totals["phantom"] / frames if frames else float("nan"),
        "buckets": totals, "errors": errs,
    }


def summarise(res):
    b = res["buckets"]
    print(f"\n  {res['detector']:>6}: recall {res['hits']}/{res['gt']} "
          f"= {100 * res['recall']:.0f}%   over {res['frames']} frames")
    print(f"          false positives {res['fp']} = "
          f"{res['fp_per_frame']:.2f}/frame  "
          f"({b['duplicate']} duplicate, {b['edge']} edge-clipped, "
          f"{b['phantom']} phantom)")
    print(f"          PHANTOM rate {res['phantom_per_frame']:.2f}/frame  "
          f"<- the one that drives the arm at nothing")
    if res["errors"]:
        e = np.array([v for _, v, _ in res["errors"]])
        n = np.array([m for _, _, m in res["errors"]])
        print(f"          position error: mean {1000 * n.mean():.1f} mm · "
              f"p95 {1000 * np.percentile(n, 95):.1f} mm · "
              f"max {1000 * n.max():.1f} mm   (n={len(n)})")
        print("          per axis, mean |err|: "
              + "  ".join(f"{a} {1000 * np.abs(e[:, i]).mean():.1f}"
                          for i, a in enumerate("xyz")) + " mm")


def main():
    import argparse

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from camera import CAM_NAME, SensorCamera
    from greenhouse import build_scene
    from mission import park_posture, reset_park
    from plant_row import Row

    ap = argparse.ArgumentParser(
        description="Score the detectors on rendered frames.")
    ap.add_argument("--camera", default=CAM_NAME)
    ap.add_argument("--only", choices=["hsv", "yolo"], default=None)
    ap.add_argument("--save", default=None, help="write overlaid frames here")
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()

    model = build_scene(wrist_cam=True)
    data = mujoco.MjData(model)
    q = park_posture(model)
    reset_park(model, data, q)
    row = Row(model, data)
    mujoco.mj_forward(model, data)

    sensor = SensorCamera(model, args.camera)
    print(f"\n  camera '{args.camera}' · {sensor.intr}")
    print("  scored on recall and false positives per frame — not mAP.")
    print("  A missed fruit is lost revenue. A false positive drives the arm")
    print("  into the canopy at a fruit that is not there.")

    dets = []
    if args.only in (None, "hsv"):
        dets.append(HSVDetector())
    if args.only in (None, "yolo"):
        try:
            dets.append(YoloDetector(conf=args.conf))
        except Exception as exc:
            print(f"\n  yolo unavailable: {exc}")

    results = []
    try:
        for d in dets:
            results.append(score_detector(model, data, sensor, d, q, row,
                                          save=args.save))
    finally:
        sensor.close()

    print(f"\n{'=' * 78}")
    for r in results:
        summarise(r)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Challenge - Snapchat Filter (Glasses + Mustache)

- Detect eyes & nose using Haar cascades
- Overlay glasses across both eyes and mustache under the nose (alpha-aware)
- Show the final image (OpenCV window)
- Save submission CSV: flatten BGR image to (-1, 3) with columns:
  ["channel 1", "channel 2", "channel 3"] (uint8)
"""

import cv2
import numpy as np
import pandas as pd

# ----------------------------- Utils -----------------------------

def overlay_png(bg_bgr, fg_rgba, x, y, w=None, h=None):
    """
    Paste fg_rgba (BGRA) onto bg_bgr (BGR) at top-left (x,y).
    Optional resize to (w,h). Handles alpha blending and clips to bounds.
    Returns modified bg_bgr (in-place safe).
    """
    if fg_rgba is None or bg_bgr is None:
        return bg_bgr

    # Ensure overlay has 4 channels (BGRA)
    if fg_rgba.shape[2] == 3:
        alpha = 255 * np.ones(fg_rgba.shape[:2], dtype=np.uint8)
        fg_rgba = np.dstack([fg_rgba, alpha])

    if w is not None and h is not None:
        fg_rgba = cv2.resize(fg_rgba, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    fh, fw = fg_rgba.shape[:2]
    H, W   = bg_bgr.shape[:2]

    # Clip to background bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + fw), min(H, y + fh)
    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    # Corresponding overlay slice
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay_rgb = fg_rgba[oy1:oy2, ox1:ox2, :3]
    alpha       = fg_rgba[oy1:oy2, ox1:ox2, 3:4].astype(float) / 255.0  # (h,w,1)

    roi = bg_bgr[y1:y2, x1:x2, :]
    bg_bgr[y1:y2, x1:x2, :] = (alpha * overlay_rgb + (1.0 - alpha) * roi).astype(np.uint8)
    return bg_bgr


def merge_two_largest_boxes(boxes):
    """
    From (x,y,w,h) detections, pick two largest by area and return a box covering both.
    If <2 detections, return the single or None.
    """
    if boxes is None or len(boxes) == 0:
        return None
    if len(boxes) == 1:
        x, y, w, h = boxes[0]
        return int(x), int(y), int(w), int(h)

    areas = sorted([(i, b[2] * b[3]) for i, b in enumerate(boxes)],
                   key=lambda t: t[1], reverse=True)
    b1 = boxes[areas[0][0]]
    b2 = boxes[areas[1][0]]

    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[0] + b1[2], b2[0] + b2[2])
    y2 = max(b1[1] + b1[3], b2[1] + b2[3])
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def pick_best_nose(noses, eye_line_y=None):
    """
    Prefer noses below the eye line (if given); otherwise choose largest area.
    """
    if noses is None or len(noses) == 0:
        return None

    candidates = [b for b in noses if eye_line_y is None or b[1] > eye_line_y]
    if not candidates:
        candidates = list(noses)

    nx, ny, nw, nh = max(candidates, key=lambda b: b[2] * b[3])
    return int(nx), int(ny), int(nw), int(nh)


# ----------------------------- Paths -----------------------------

IMG_PATH      = '.\snap_filter_Train\Test\Before.png'
GLASSES_PATH  = './snap_filter_Train/glasses.png'
MUSTACHE_PATH = './snap_filter_Train/mustache.png'
EYES_CASCADE  = './snap_filter_Train/third-party/frontalEyes35x16.xml'
NOSE_CASCADE  = './snap_filter_Train/third-party/Nose18x15.xml'

# ----------------------------- Load ------------------------------

img = cv2.imread(IMG_PATH)  # BGR
if img is None:
    raise FileNotFoundError(f"Could not read image at {IMG_PATH}")

# Lock original shape & dtype
H0, W0 = img.shape[:2]
dtype0 = img.dtype

# Load filters with alpha preserved
glasses_img  = cv2.imread(GLASSES_PATH,  cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread(MUSTACHE_PATH, cv2.IMREAD_UNCHANGED)
if glasses_img is None:
    raise FileNotFoundError(f"Could not read glasses at {GLASSES_PATH}")
if mustache_img is None:
    raise FileNotFoundError(f"Could not read mustache at {MUSTACHE_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)  # improves cascade robustness

eyes_cascade = cv2.CascadeClassifier(EYES_CASCADE)
nose_cascade = cv2.CascadeClassifier(NOSE_CASCADE)
if eyes_cascade.empty() or nose_cascade.empty():
    raise RuntimeError("Failed to load cascades. Check paths.")

# --------------------------- Detection ---------------------------

eyes  = eyes_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(20, 20))
noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(20, 20))

eye_line_y = None
if len(eyes) > 0:
    centers_y = [ey + eh // 2 for (_, ey, _, eh) in eyes]
    eye_line_y = int(np.mean(centers_y))

# ------------------------ Glasses placement ----------------------

gbox = merge_two_largest_boxes(eyes)
if gbox is not None:
    gx, gy, gw, gh = gbox
    # slight expansion for nicer fit
    sw, sh = 1.15, 1.20
    new_w, new_h = int(gw * sw), int(gh * sh)
    gx -= (new_w - gw) // 2
    gy -= (new_h - gh) // 2
    img = overlay_png(img, glasses_img, gx, gy, new_w, new_h)

# ----------------------- Mustache placement ----------------------

nbox = pick_best_nose(noses, eye_line_y)
if nbox is not None:
    nx, ny, nw, nh = nbox
    target_w = int(1.20 * nw)
    ar       = mustache_img.shape[0] / mustache_img.shape[1]  # h/w
    target_h = max(1, int(target_w * ar))
    mx = nx + (nw - target_w) // 2
    my = ny + int(0.45 * nh)
    img = overlay_png(img, mustache_img, mx, my, target_w, target_h)

# --------------------- Safety: shape/dtype -----------------------

assert img.shape[:2] == (H0, W0), f"Image size changed! Was {(H0,W0)}, got {img.shape[:2]}"
assert img.dtype == dtype0,       f"Image dtype changed! Was {dtype0}, got {img.dtype}"

# --------------------------- Show result -------------------------

cv2.imshow("Snap Filter Result", img)
cv2.waitKey(3000)
cv2.destroyAllWindows()

# ----------------------- Save submission CSV ---------------------

flat_bgr = img.reshape(H0 * W0, 3)  # BGR row-major
df = pd.DataFrame(flat_bgr, columns=["Channel 1", "Channel 2", "Channel 3"])
df = df.astype({"Channel 1": "int64", "Channel 2": "int64", "Channel 3": "int64"})
df.to_csv("submission.csv", index=False)

print(
    f"Done. Image size: {W0}x{H0}. "
    f"CSV saved as submission.csv with shape {df.shape} and dtypes:\n{df.dtypes}"
)

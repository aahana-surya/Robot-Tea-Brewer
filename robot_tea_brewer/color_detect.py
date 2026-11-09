"""Simple HSV color detection for the green tea-bag tag."""
import cv2
import numpy as np

# OpenCV HSV: H 0-179, S/V 0-255
GREEN = ((45, 120, 80), (85, 255, 255))   # tag rgba .1 .7 .2


def _blobs(frame_bgr, lo, hi, min_area, max_area):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        fill = area / float(w * h)            # 1.0 for a filled rectangle
        circ = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-6)
        out.append({'bbox': [x, y, x + w, y + h], 'area': area,
                    'fill': fill, 'circ': circ})
    return out


def detect_tag(frame_bgr, min_area=6, max_area=5000):
    """Find the green tag. The scene has no other green object.

    Returns {bbox, center, area} for the largest rectangular green blob, or None.
    """
    blobs = [b for b in _blobs(frame_bgr, *GREEN, min_area, max_area)
             if b['fill'] > 0.4]
    if not blobs:
        return None
    b = max(blobs, key=lambda b: b['area'])
    x1, y1, x2, y2 = b['bbox']
    return {'bbox': b['bbox'], 'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            'area': b['area']}

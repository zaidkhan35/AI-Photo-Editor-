# AI Photo Editor — Phase 1
# Basic Editor: crop, resize, rotate, flip, brightness, contrast, blur, sharpen, grayscale

import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────
# LOAD IMAGE
# ──────────────────────────────
def load_default_image(path=None):
    global img, original, h, w

    if path is None:
        path = PROJECT_ROOT / "test.png"

    img = cv2.imread(str(path))
    if img is None:
        print(f"Warning: default image not found at {path}.")
        original = None
        h = 0
        w = 0
        return None

    original = img.copy()
    h, w = img.shape[:2]
    return original

img = None
original = None
h = 0
w = 0
load_default_image()


# PIPELINE STATE
# inital values for the trackbars

state = {
    "brightness": 100,
    "contrast":   10,
    "blur":       1,
    "sharpen":    0,
    "grayscale":  0,
    "scale":      100,
    "rotate":     0,
    "flip":       0,
    "crop_x":     0,
    "crop_y":     0,
    "crop_w":     w,
    "crop_h":     h,
}

def nothing(x):
    pass


# OPERATIONS

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_brightness(image, val):
    actual   = val - 100
    hsv      = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hh, s, v = cv2.split(hsv)
    v        = cv2.add(v, actual)
    return cv2.cvtColor(cv2.merge([hh, s, v]), cv2.COLOR_HSV2BGR)

def apply_contrast(image, val):
    alpha = max(val / 10.0, 0.1)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def apply_blur(image, val):
    if val % 2 == 0: val += 1
    if val < 1:      val  = 1
    return cv2.GaussianBlur(image, (val, val), 0)

def apply_sharpen(image):
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return cv2.filter2D(image, -1, kernel)

def apply_resize(image, scale):
    ih, iw = image.shape[:2]
    new_w  = max(1, int(iw * scale / 100))
    new_h  = max(1, int(ih * scale / 100))
    return cv2.resize(image, (new_w, new_h))

def apply_rotate(image, angle):
    if angle == 0: return image
    ih, iw = image.shape[:2]
    center = (iw // 2, ih // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(image, matrix, (iw, ih))

def apply_flip(image, mode):
    if   mode == 0: return image
    elif mode == 1: return cv2.flip(image, 1)
    elif mode == 2: return cv2.flip(image, 0)
    elif mode == 3: return cv2.flip(image, -1)

def apply_crop(image):
    ih, iw = image.shape[:2]
    x  = min(state["crop_x"], iw - 1)
    y  = min(state["crop_y"], ih - 1)
    cw = min(state["crop_w"], iw - x)
    ch = min(state["crop_h"], ih - y)
    if cw < 1: cw = 1
    if ch < 1: ch = 1
    return image[y:y+ch, x:x+cw]


# PIPELINE

def run_pipeline():
    if original is None:
        return None

    result = original.copy()

    result = apply_crop(result)
    result = apply_resize(result,     state["scale"])
    result = apply_rotate(result,     state["rotate"])
    result = apply_flip(result,       state["flip"])
    result = apply_brightness(result, state["brightness"])
    result = apply_contrast(result,   state["contrast"])

    if state["blur"] > 1:
        result = apply_blur(result, state["blur"])

    if state["sharpen"] == 1:
        result = apply_sharpen(result)

    if state["grayscale"] == 1:
        result = apply_grayscale(result)

    return result


# PREVIEW

def show_preview(result, win):
    if original is None or result is None:
        return

    display_h = 400
    oh, ow    = original.shape[:2]
    rh, rw    = result.shape[:2]

    left  = cv2.resize(original, (int(ow * display_h / oh), display_h))
    right = cv2.resize(result,   (int(rw * display_h / rh), display_h))

    cv2.putText(left,  "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(right, "RESULT",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    divider    = np.zeros((display_h, 3, 3), dtype=np.uint8)
    divider[:] = (0, 255, 0)

    cv2.imshow(win, np.hstack([left, divider, right]))


# MAIN — only runs when called directly

if __name__ == "__main__":

    WIN = "Basic Editor  |  S=Save  R=Reset  Q=Quit"
    cv2.namedWindow(WIN)

    cv2.createTrackbar("Grayscale 0/1", WIN, 0,   1,   nothing)
    cv2.createTrackbar("Brightness",    WIN, 100, 200, nothing)
    cv2.createTrackbar("Contrast",      WIN, 10,  40,  nothing)
    cv2.createTrackbar("Blur",          WIN, 1,   51,  nothing)
    cv2.createTrackbar("Sharpen  0/1",  WIN, 0,   1,   nothing)
    cv2.createTrackbar("Scale %",       WIN, 100, 300, nothing)
    cv2.createTrackbar("Rotate",        WIN, 0,   360, nothing)
    cv2.createTrackbar("Flip  0-3",     WIN, 0,   3,   nothing)
    cv2.createTrackbar("Crop X",        WIN, 0,   w-1, nothing)
    cv2.createTrackbar("Crop Y",        WIN, 0,   h-1, nothing)
    cv2.createTrackbar("Crop W",        WIN, w,   w,   nothing)
    cv2.createTrackbar("Crop H",        WIN, h,   h,   nothing)

    print("\n--- AI Photo Editor | Phase 1 ---")
    print("Grayscale: 0=color  1=gray")
    print("Sharpen:   0=off    1=on")
    print("Flip:      0=none   1=horizontal  2=vertical  3=both")
    print("S=Save  |  R=Reset  |  Q=Quit")
# createTrackbar  →  creates the slider
# getTrackbarPos  →  reads the slider value

    while True:
        state["grayscale"]  = cv2.getTrackbarPos("Grayscale 0/1", WIN)
        state["brightness"] = cv2.getTrackbarPos("Brightness",    WIN)
        state["contrast"]   = cv2.getTrackbarPos("Contrast",      WIN)
        state["blur"]       = cv2.getTrackbarPos("Blur",          WIN)
        state["sharpen"]    = cv2.getTrackbarPos("Sharpen  0/1",  WIN)
        state["scale"]      = cv2.getTrackbarPos("Scale %",       WIN)
        state["rotate"]     = cv2.getTrackbarPos("Rotate",        WIN)
        state["flip"]       = cv2.getTrackbarPos("Flip  0-3",     WIN)
        state["crop_x"]     = cv2.getTrackbarPos("Crop X",        WIN)
        state["crop_y"]     = cv2.getTrackbarPos("Crop Y",        WIN)
        state["crop_w"]     = cv2.getTrackbarPos("Crop W",        WIN)
        state["crop_h"]     = cv2.getTrackbarPos("Crop H",        WIN)

        result = run_pipeline()
        show_preview(result, WIN)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite("output_phase1.jpg", result)
            print("Saved as output_phase1.jpg")

        elif key == ord('r'):
            cv2.setTrackbarPos("Grayscale 0/1", WIN, 0)
            cv2.setTrackbarPos("Brightness",    WIN, 100)
            cv2.setTrackbarPos("Contrast",      WIN, 10)
            cv2.setTrackbarPos("Blur",          WIN, 1)
            cv2.setTrackbarPos("Sharpen  0/1",  WIN, 0)
            cv2.setTrackbarPos("Scale %",       WIN, 100)
            cv2.setTrackbarPos("Rotate",        WIN, 0)
            cv2.setTrackbarPos("Flip  0-3",     WIN, 0)
            cv2.setTrackbarPos("Crop X",        WIN, 0)
            cv2.setTrackbarPos("Crop Y",        WIN, 0)
            cv2.setTrackbarPos("Crop W",        WIN, w)
            cv2.setTrackbarPos("Crop H",        WIN, h)
            print("Reset to original")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
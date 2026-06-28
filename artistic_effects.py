
# Artistic Effects: cartoon, sketch, color sketch, stylization, edges, emboss, sepia
#  we are Importing  Phase 1 pipeline from basic_editor.py

import cv2
import numpy as np
from basic_editor import original, h, w, state, run_pipeline, show_preview, nothing


state["cartoon"]      = 0
state["sketch_bw"]    = 0
state["sketch_color"] = 0
state["stylize"]      = 0
state["edges"]        = 0
state["emboss"]       = 0
state["sepia"]        = 0


def apply_cartoon(image, strength):
    sigma = 25 + int(strength * 0.5)
    color = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    low   = max(10, int(strength * 1.0))
    high  = max(50, int(strength * 2.0))
    edges = cv2.Canny(blur, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, 255 - edges)

def apply_sketch_bw(image, strength):
    ksize = int(strength * 0.25) * 2 + 1
    if ksize < 1: ksize = 1
    gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted      = cv2.bitwise_not(gray)
    blurred       = cv2.GaussianBlur(inverted, (ksize, ksize), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch        = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_sketch_color(image, strength):
    ksize         = 21
    gray          = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted      = cv2.bitwise_not(gray)
    blurred       = cv2.GaussianBlur(inverted, (ksize, ksize), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch        = cv2.divide(gray, inverted_blur, scale=256.0)
    sketch_bgr    = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    alpha         = strength / 100.0
    return cv2.addWeighted(image, alpha, sketch_bgr, 1 - alpha, 0)

def apply_stylization(image, strength):
    sigma_s = 1 + int(strength * 0.99)
    sigma_r = 0.01 + strength * 0.0049
    return cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)

def apply_edges(image, strength):
    low   = max(1, int(strength * 1.0))
    high  = low * 2
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, low, high)
    edges = cv2.bitwise_not(edges)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_emboss(image, strength):
    depth  = strength / 50.0
    kernel = np.array([
        [-2 * depth, -depth,       0],
        [-depth,      1,           depth],
        [ 0,          depth, 2 * depth]
    ])
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    emboss = cv2.filter2D(gray, -1, kernel)
    emboss = np.clip(emboss + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

def apply_sepia(image, strength):
    factor       = strength / 100.0
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia = cv2.transform(image, sepia_matrix)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.addWeighted(image, 1 - factor, sepia, factor, 0)


def run_full_pipeline():
    result = run_pipeline()

    if state["cartoon"]      > 0: result = apply_cartoon(result,      state["cartoon"])
    if state["sketch_bw"]    > 0: result = apply_sketch_bw(result,    state["sketch_bw"])
    if state["sketch_color"] > 0: result = apply_sketch_color(result, state["sketch_color"])
    if state["stylize"]      > 0: result = apply_stylization(result,  state["stylize"])
    if state["edges"]        > 0: result = apply_edges(result,        state["edges"])
    if state["emboss"]       > 0: result = apply_emboss(result,       state["emboss"])
    if state["sepia"]        > 0: result = apply_sepia(result,        state["sepia"])

    return result


if __name__ == "__main__":

    WIN = "Artistic Effects  |  S=Save  R=Reset  Q=Quit"
    cv2.namedWindow(WIN)

    # ── phase 1 sliders ──
    cv2.createTrackbar("Grayscale 0/1",    WIN, 0,   1,   nothing)
    cv2.createTrackbar("Brightness",       WIN, 100, 200, nothing)
    cv2.createTrackbar("Contrast",         WIN, 10,  40,  nothing)
    cv2.createTrackbar("Blur",             WIN, 1,   51,  nothing)
    cv2.createTrackbar("Sharpen  0/1",     WIN, 0,   1,   nothing)
    cv2.createTrackbar("Scale %",          WIN, 100, 300, nothing)
    cv2.createTrackbar("Rotate",           WIN, 0,   360, nothing)
    cv2.createTrackbar("Flip  0-3",        WIN, 0,   3,   nothing)
    cv2.createTrackbar("Crop X",           WIN, 0,   w-1, nothing)
    cv2.createTrackbar("Crop Y",           WIN, 0,   h-1, nothing)
    cv2.createTrackbar("Crop W",           WIN, w,   w,   nothing)
    cv2.createTrackbar("Crop H",           WIN, h,   h,   nothing)

    # ── phase 2 sliders — 0=off, 1-100=strength ──
    cv2.createTrackbar("Cartoon   0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch BW 0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch CL 0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Stylize   0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Edges     0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Emboss    0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sepia     0-100",  WIN, 0, 100, nothing)

    print("\n--- AI Photo Editor | Phase 2 ---")
    print("0   = effect off")
    print("1-100 = effect on, higher = stronger")
    print("multiple effects can be on at same time")
    print("S=Save  |  R=Reset  |  Q=Quit")

    while True:

        # phase 1
        state["grayscale"]    = cv2.getTrackbarPos("Grayscale 0/1",   WIN)
        state["brightness"]   = cv2.getTrackbarPos("Brightness",      WIN)
        state["contrast"]     = cv2.getTrackbarPos("Contrast",        WIN)
        state["blur"]         = cv2.getTrackbarPos("Blur",            WIN)
        state["sharpen"]      = cv2.getTrackbarPos("Sharpen  0/1",    WIN)
        state["scale"]        = cv2.getTrackbarPos("Scale %",         WIN)
        state["rotate"]       = cv2.getTrackbarPos("Rotate",          WIN)
        state["flip"]         = cv2.getTrackbarPos("Flip  0-3",       WIN)
        state["crop_x"]       = cv2.getTrackbarPos("Crop X",          WIN)
        state["crop_y"]       = cv2.getTrackbarPos("Crop Y",          WIN)
        state["crop_w"]       = cv2.getTrackbarPos("Crop W",          WIN)
        state["crop_h"]       = cv2.getTrackbarPos("Crop H",          WIN)

        # phase 2
        state["cartoon"]      = cv2.getTrackbarPos("Cartoon   0-100", WIN)
        state["sketch_bw"]    = cv2.getTrackbarPos("Sketch BW 0-100", WIN)
        state["sketch_color"] = cv2.getTrackbarPos("Sketch CL 0-100", WIN)
        state["stylize"]      = cv2.getTrackbarPos("Stylize   0-100", WIN)
        state["edges"]        = cv2.getTrackbarPos("Edges     0-100", WIN)
        state["emboss"]       = cv2.getTrackbarPos("Emboss    0-100", WIN)
        state["sepia"]        = cv2.getTrackbarPos("Sepia     0-100", WIN)

        result = run_full_pipeline()
        show_preview(result, WIN)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite("output_phase2.jpg", result)
            print("Saved as output_phase2.jpg")

        elif key == ord('r'):
            cv2.setTrackbarPos("Grayscale 0/1",   WIN, 0)
            cv2.setTrackbarPos("Brightness",      WIN, 100)
            cv2.setTrackbarPos("Contrast",        WIN, 10)
            cv2.setTrackbarPos("Blur",            WIN, 1)
            cv2.setTrackbarPos("Sharpen  0/1",    WIN, 0)
            cv2.setTrackbarPos("Scale %",         WIN, 100)
            cv2.setTrackbarPos("Rotate",          WIN, 0)
            cv2.setTrackbarPos("Flip  0-3",       WIN, 0)
            cv2.setTrackbarPos("Crop X",          WIN, 0)
            cv2.setTrackbarPos("Crop Y",          WIN, 0)
            cv2.setTrackbarPos("Crop W",          WIN, w)
            cv2.setTrackbarPos("Crop H",          WIN, h)
            cv2.setTrackbarPos("Cartoon   0-100", WIN, 0)
            cv2.setTrackbarPos("Sketch BW 0-100", WIN, 0)
            cv2.setTrackbarPos("Sketch CL 0-100", WIN, 0)
            cv2.setTrackbarPos("Stylize   0-100", WIN, 0)
            cv2.setTrackbarPos("Edges     0-100", WIN, 0)
            cv2.setTrackbarPos("Emboss    0-100", WIN, 0)
            cv2.setTrackbarPos("Sepia     0-100", WIN, 0)
            print("Reset to defaults")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
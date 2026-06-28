
# Face Detection: detect faces, count, bounding boxes, blur, pixelate, highlight
# Imports Phase 1 pipeline from basic_editor.py
# Imports Phase 2 pipeline from artistic_effects.py

import cv2
import numpy as np
from basic_editor     import original, h, w, state, show_preview, nothing
from artistic_effects import run_full_pipeline


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


state["face_mode"]         = 0
state["min_neighbors"]     = 5
state["scale_factor_x10"]  = 12    # changed from 11 to 12 to avoid 1.0 crash


def detect_faces(image):
    gray         = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_factor = state["scale_factor_x10"] / 10.0

    # fix — must always be greater than 1.0
    if scale_factor <= 1.0:
        scale_factor = 1.1

    min_neighbors = max(1, state["min_neighbors"])

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor  = scale_factor,
        minNeighbors = min_neighbors,
        minSize      = (30, 30)
    )
    return faces if len(faces) > 0 else []


def draw_boxes(image, faces):
    result = image.copy()
    for (x, y, fw, fh) in faces:
        cv2.rectangle(result, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
    return result


def blur_faces(image, faces):
    result = image.copy()
    for (x, y, fw, fh) in faces:
        roi                     = result[y:y+fh, x:x+fw]
        roi_blurred             = cv2.GaussianBlur(roi, (51, 51), 30)
        result[y:y+fh, x:x+fw] = roi_blurred
    return result



def pixelate_faces(image, faces):
    result = image.copy()
    for (x, y, fw, fh) in faces:
        roi       = result[y:y+fh, x:x+fw]
        pixel_size = 10
        small     = cv2.resize(roi, (pixel_size, pixel_size),
                               interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (fw, fh),
                               interpolation=cv2.INTER_NEAREST)
        result[y:y+fh, x:x+fw] = pixelated
    return result



def highlight_faces(image, faces):
    darkened = cv2.convertScaleAbs(image, alpha=0.3, beta=0)
    result   = darkened.copy()
    for (x, y, fw, fh) in faces:
        result[y:y+fh, x:x+fw] = image[y:y+fh, x:x+fw]
        cv2.rectangle(result, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
    return result

def draw_face_count(image, count):
    result = image.copy()
    cv2.putText(result, f"Faces: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    return result


def run_phase3_pipeline():
    result = run_full_pipeline()

    faces  = detect_faces(result)
    count  = len(faces)
    mode   = state["face_mode"]

    if   mode == 1: result = draw_boxes(result,      faces)
    elif mode == 2: result = blur_faces(result,       faces)
    elif mode == 3: result = pixelate_faces(result,   faces)
    elif mode == 4: result = highlight_faces(result,  faces)

    if mode > 0:
        result = draw_face_count(result, count)
        print(f"Faces detected: {count}", end="\r")

    return result


if __name__ == "__main__":

    WIN = "Face Detection  |  S=Save  R=Reset  Q=Quit"
    cv2.namedWindow(WIN)

    # phase 1 sliders
    cv2.createTrackbar("Grayscale 0/1",   WIN, 0,   1,   nothing)
    cv2.createTrackbar("Brightness",      WIN, 100, 200, nothing)
    cv2.createTrackbar("Contrast",        WIN, 10,  40,  nothing)
    cv2.createTrackbar("Blur",            WIN, 1,   51,  nothing)
    cv2.createTrackbar("Sharpen  0/1",    WIN, 0,   1,   nothing)
    cv2.createTrackbar("Scale %",         WIN, 100, 300, nothing)
    cv2.createTrackbar("Rotate",          WIN, 0,   360, nothing)
    cv2.createTrackbar("Flip  0-3",       WIN, 0,   3,   nothing)
    cv2.createTrackbar("Crop X",          WIN, 0,   w-1, nothing)
    cv2.createTrackbar("Crop Y",          WIN, 0,   h-1, nothing)
    cv2.createTrackbar("Crop W",          WIN, w,   w,   nothing)
    cv2.createTrackbar("Crop H",          WIN, h,   h,   nothing)

    # phase 2 sliders
    cv2.createTrackbar("Cartoon   0-100", WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch BW 0-100", WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch CL 0-100", WIN, 0, 100, nothing)
    cv2.createTrackbar("Stylize   0-100", WIN, 0, 100, nothing)
    cv2.createTrackbar("Edges     0-100", WIN, 0, 100, nothing)
    cv2.createTrackbar("Emboss    0-100", WIN, 0, 100, nothing)
    cv2.createTrackbar("Sepia     0-100", WIN, 0, 100, nothing)

    # phase 3 sliders
    cv2.createTrackbar("Face Mode  0-4",  WIN, 0,  4,  nothing)
    cv2.createTrackbar("Min Neighbors",   WIN, 5,  20, nothing)
    cv2.createTrackbar("Scale x10",       WIN, 12, 20, nothing)  # default 12 not 11

    print("\n--- AI Photo Editor | Phase 3 ---")
    print("Face Mode:")
    print("  0 = off")
    print("  1 = bounding boxes")
    print("  2 = blur faces")
    print("  3 = pixelate faces")
    print("  4 = highlight faces")
    print("")
    print("Min Neighbors — higher = stricter, fewer false detections")
    print("Scale x10     — 12=1.2  15=1.5  (must stay above 10)")
    print("S=Save  |  R=Reset  |  Q=Quit")

    while True:

        # phase 1
        state["grayscale"]        = cv2.getTrackbarPos("Grayscale 0/1",   WIN)
        state["brightness"]       = cv2.getTrackbarPos("Brightness",      WIN)
        state["contrast"]         = cv2.getTrackbarPos("Contrast",        WIN)
        state["blur"]             = cv2.getTrackbarPos("Blur",            WIN)
        state["sharpen"]          = cv2.getTrackbarPos("Sharpen  0/1",    WIN)
        state["scale"]            = cv2.getTrackbarPos("Scale %",         WIN)
        state["rotate"]           = cv2.getTrackbarPos("Rotate",          WIN)
        state["flip"]             = cv2.getTrackbarPos("Flip  0-3",       WIN)
        state["crop_x"]           = cv2.getTrackbarPos("Crop X",          WIN)
        state["crop_y"]           = cv2.getTrackbarPos("Crop Y",          WIN)
        state["crop_w"]           = cv2.getTrackbarPos("Crop W",          WIN)
        state["crop_h"]           = cv2.getTrackbarPos("Crop H",          WIN)

        # phase 2
        state["cartoon"]          = cv2.getTrackbarPos("Cartoon   0-100", WIN)
        state["sketch_bw"]        = cv2.getTrackbarPos("Sketch BW 0-100", WIN)
        state["sketch_color"]     = cv2.getTrackbarPos("Sketch CL 0-100", WIN)
        state["stylize"]          = cv2.getTrackbarPos("Stylize   0-100", WIN)
        state["edges"]            = cv2.getTrackbarPos("Edges     0-100", WIN)
        state["emboss"]           = cv2.getTrackbarPos("Emboss    0-100", WIN)
        state["sepia"]            = cv2.getTrackbarPos("Sepia     0-100", WIN)

        # phase 3
        state["face_mode"]        = cv2.getTrackbarPos("Face Mode  0-4",  WIN)
        state["min_neighbors"]    = cv2.getTrackbarPos("Min Neighbors",   WIN)
        state["scale_factor_x10"] = cv2.getTrackbarPos("Scale x10",       WIN)

        result = run_phase3_pipeline()
        show_preview(result, WIN)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite("output_phase3.jpg", result)
            print("\nSaved as output_phase3.jpg")

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
            cv2.setTrackbarPos("Face Mode  0-4",  WIN, 0)
            cv2.setTrackbarPos("Min Neighbors",   WIN, 5)
            cv2.setTrackbarPos("Scale x10",       WIN, 12)  # reset to 12
            print("\nReset to defaults")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

# Advanced Features: text overlay, quality enhancement, image compression
# Imports all previous pipelines

import cv2
import numpy as np
from PIL              import Image, ImageDraw, ImageFont
from basic_editor     import original, h, w, state, show_preview, nothing
from artistic_effects import run_full_pipeline
from face_detection   import run_phase3_pipeline
from ai_features      import run_phase4_pipeline


state["text_on"]      = 0
state["text_content"] = "Your Text Here"
state["text_x"]       = 10
state["text_y"]       = 10
state["text_size"]    = 40
state["text_r"]       = 255
state["text_g"]       = 255
state["text_b"]       = 255
state["enhance_qual"] = 0
state["compress"]     = 95


def apply_text(image):
    if state["text_content"].strip() == "":
        return image

    # convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)

    # load font — fall back to default if arial not found
    try:
        font = ImageFont.truetype("arial.ttf", state["text_size"])
    except Exception:
        try:
            # try common system font locations
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", state["text_size"])
        except Exception:
            font = ImageFont.load_default()

    # position from sliders as percentage of image size
    ih, iw = image.shape[:2]
    x      = int(state["text_x"] / 100 * iw)
    y      = int(state["text_y"] / 100 * ih)

    # color from R G B sliders
    color  = (state["text_r"], state["text_g"], state["text_b"])

    # draw text with thin black outline for visibility on any background
    outline_color = (0, 0, 0)
    draw.text((x-1, y-1), state["text_content"], font=font, fill=outline_color)
    draw.text((x+1, y-1), state["text_content"], font=font, fill=outline_color)
    draw.text((x-1, y+1), state["text_content"], font=font, fill=outline_color)
    draw.text((x+1, y+1), state["text_content"], font=font, fill=outline_color)
    draw.text((x,   y),   state["text_content"], font=font, fill=color)

    # convert back to BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def apply_quality_enhance(image, strength):
    # step 1 — create blurred version
    blur_amount = max(1, int(strength * 0.1))
    if blur_amount % 2 == 0:
        blur_amount += 1
    blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # step 2 — unsharp mask
    # original + (original - blurred) * amount
    amount  = strength / 50.0
    result  = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

    # step 3 — CLAHE for local contrast boost
    lab     = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge([l, a, b])
    result  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


def apply_compression(image, quality):
    if quality < 1:
        quality = 1

    # encode to jpeg in memory at given quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded   = cv2.imencode(".jpg", image, encode_param)

    # decode back to image array
    result       = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # show size comparison in terminal
    original_kb  = image.nbytes / 1024
    compressed_kb= len(encoded) / 1024
    print(f"Size — Original: {original_kb:.1f} KB  |  "
          f"Compressed: {compressed_kb:.1f} KB  |  "
          f"Quality: {quality}%", end="\r")

    return result


def run_phase5_pipeline():
    result = run_phase4_pipeline()   # phases 1+2+3+4

    if state["enhance_qual"] > 0:
        result = apply_quality_enhance(result, state["enhance_qual"])

    if state["text_on"] == 1:
        result = apply_text(result)

    # compression always last
    if state["compress"] < 95:
        result = apply_compression(result, state["compress"])

    return result


if __name__ == "__main__":

    WIN = "Advanced Features  |  S=Save  T=Type Text  R=Reset  Q=Quit"
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

    # ── phase 2 sliders ──
    cv2.createTrackbar("Cartoon   0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch BW 0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch CL 0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Stylize   0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Edges     0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Emboss    0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sepia     0-100",  WIN, 0, 100, nothing)

    # ── phase 3 sliders ──
    cv2.createTrackbar("Face Mode  0-4",   WIN, 0,  4,  nothing)
    cv2.createTrackbar("Min Neighbors",    WIN, 5,  20, nothing)
    cv2.createTrackbar("Scale x10",        WIN, 12, 20, nothing)

    # ── phase 4 sliders ──
    cv2.createTrackbar("BG Remove  0/1",   WIN, 0,  1,   nothing)
    cv2.createTrackbar("Denoise    0-100", WIN, 0,  100, nothing)
    cv2.createTrackbar("Enhance    0-100", WIN, 0,  100, nothing)
    cv2.createTrackbar("Super Res  0-100", WIN, 0,  100, nothing)
    cv2.createTrackbar("Color Trnsfr 0/1", WIN, 0,  1,   nothing)

    # ── phase 5 sliders ──
    cv2.createTrackbar("Text       0/1",   WIN, 0,   1,   nothing)
    cv2.createTrackbar("Text X %",         WIN, 10,  100, nothing)
    cv2.createTrackbar("Text Y %",         WIN, 10,  100, nothing)
    cv2.createTrackbar("Text Size",        WIN, 40,  200, nothing)
    cv2.createTrackbar("Text R",           WIN, 255, 255, nothing)
    cv2.createTrackbar("Text G",           WIN, 255, 255, nothing)
    cv2.createTrackbar("Text B",           WIN, 255, 255, nothing)
    cv2.createTrackbar("Qual Enh  0-100",  WIN, 0,   100, nothing)
    cv2.createTrackbar("Compress  1-100",  WIN, 95,  100, nothing)

    print("\n--- AI Photo Editor | Phase 5 ---")
    print("Text       — turn on then press T to type your text")
    print("Text X/Y % — position on image as percentage")
    print("Text Size  — font size 1 to 200")
    print("Text R/G/B — text color (255,255,255 = white)")
    print("Qual Enh   — professional sharpening + contrast boost")
    print("Compress   — 95=best quality  1=smallest file size")
    print("             watch terminal for KB size comparison")
    print("")
    print("S=Save  |  T=Type text  |  R=Reset  |  Q=Quit")

    while True:

        # phase 1
        state["grayscale"]        = cv2.getTrackbarPos("Grayscale 0/1",    WIN)
        state["brightness"]       = cv2.getTrackbarPos("Brightness",       WIN)
        state["contrast"]         = cv2.getTrackbarPos("Contrast",         WIN)
        state["blur"]             = cv2.getTrackbarPos("Blur",             WIN)
        state["sharpen"]          = cv2.getTrackbarPos("Sharpen  0/1",     WIN)
        state["scale"]            = cv2.getTrackbarPos("Scale %",          WIN)
        state["rotate"]           = cv2.getTrackbarPos("Rotate",           WIN)
        state["flip"]             = cv2.getTrackbarPos("Flip  0-3",        WIN)
        state["crop_x"]           = cv2.getTrackbarPos("Crop X",           WIN)
        state["crop_y"]           = cv2.getTrackbarPos("Crop Y",           WIN)
        state["crop_w"]           = cv2.getTrackbarPos("Crop W",           WIN)
        state["crop_h"]           = cv2.getTrackbarPos("Crop H",           WIN)

        # phase 2
        state["cartoon"]          = cv2.getTrackbarPos("Cartoon   0-100",  WIN)
        state["sketch_bw"]        = cv2.getTrackbarPos("Sketch BW 0-100",  WIN)
        state["sketch_color"]     = cv2.getTrackbarPos("Sketch CL 0-100",  WIN)
        state["stylize"]          = cv2.getTrackbarPos("Stylize   0-100",  WIN)
        state["edges"]            = cv2.getTrackbarPos("Edges     0-100",  WIN)
        state["emboss"]           = cv2.getTrackbarPos("Emboss    0-100",  WIN)
        state["sepia"]            = cv2.getTrackbarPos("Sepia     0-100",  WIN)

        # phase 3
        state["face_mode"]        = cv2.getTrackbarPos("Face Mode  0-4",   WIN)
        state["min_neighbors"]    = cv2.getTrackbarPos("Min Neighbors",    WIN)
        state["scale_factor_x10"] = cv2.getTrackbarPos("Scale x10",        WIN)

        # phase 4
        state["bg_remove"]        = cv2.getTrackbarPos("BG Remove  0/1",   WIN)
        state["denoise"]          = cv2.getTrackbarPos("Denoise    0-100",  WIN)
        state["auto_enhance"]     = cv2.getTrackbarPos("Enhance    0-100",  WIN)
        state["super_res"]        = cv2.getTrackbarPos("Super Res  0-100",  WIN)
        state["color_transfer"]   = cv2.getTrackbarPos("Color Trnsfr 0/1",  WIN)

        # phase 5
        state["text_on"]          = cv2.getTrackbarPos("Text       0/1",   WIN)
        state["text_x"]           = cv2.getTrackbarPos("Text X %",         WIN)
        state["text_y"]           = cv2.getTrackbarPos("Text Y %",         WIN)
        state["text_size"]        = cv2.getTrackbarPos("Text Size",        WIN)
        state["text_r"]           = cv2.getTrackbarPos("Text R",           WIN)
        state["text_g"]           = cv2.getTrackbarPos("Text G",           WIN)
        state["text_b"]           = cv2.getTrackbarPos("Text B",           WIN)
        state["enhance_qual"]     = cv2.getTrackbarPos("Qual Enh  0-100",  WIN)
        state["compress"]         = cv2.getTrackbarPos("Compress  1-100",  WIN)

        result = run_phase5_pipeline()
        show_preview(result, WIN)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            quality      = state["compress"]
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            cv2.imwrite("output_phase5.jpg", result, encode_param)
            print(f"\nSaved as output_phase5.jpg at quality {quality}%")

        elif key == ord('t'):
            text = input("\nType your text: ").strip()
            if text:
                state["text_content"] = text
                print(f"Text set to: {text}")
                print("Now turn Text slider to 1 to see it on image")

        elif key == ord('r'):
            # phase 1
            cv2.setTrackbarPos("Grayscale 0/1",    WIN, 0)
            cv2.setTrackbarPos("Brightness",       WIN, 100)
            cv2.setTrackbarPos("Contrast",         WIN, 10)
            cv2.setTrackbarPos("Blur",             WIN, 1)
            cv2.setTrackbarPos("Sharpen  0/1",     WIN, 0)
            cv2.setTrackbarPos("Scale %",          WIN, 100)
            cv2.setTrackbarPos("Rotate",           WIN, 0)
            cv2.setTrackbarPos("Flip  0-3",        WIN, 0)
            cv2.setTrackbarPos("Crop X",           WIN, 0)
            cv2.setTrackbarPos("Crop Y",           WIN, 0)
            cv2.setTrackbarPos("Crop W",           WIN, w)
            cv2.setTrackbarPos("Crop H",           WIN, h)
            # phase 2
            cv2.setTrackbarPos("Cartoon   0-100",  WIN, 0)
            cv2.setTrackbarPos("Sketch BW 0-100",  WIN, 0)
            cv2.setTrackbarPos("Sketch CL 0-100",  WIN, 0)
            cv2.setTrackbarPos("Stylize   0-100",  WIN, 0)
            cv2.setTrackbarPos("Edges     0-100",  WIN, 0)
            cv2.setTrackbarPos("Emboss    0-100",  WIN, 0)
            cv2.setTrackbarPos("Sepia     0-100",  WIN, 0)
            # phase 3
            cv2.setTrackbarPos("Face Mode  0-4",   WIN, 0)
            cv2.setTrackbarPos("Min Neighbors",    WIN, 5)
            cv2.setTrackbarPos("Scale x10",        WIN, 12)
            # phase 4
            cv2.setTrackbarPos("BG Remove  0/1",   WIN, 0)
            cv2.setTrackbarPos("Denoise    0-100", WIN, 0)
            cv2.setTrackbarPos("Enhance    0-100", WIN, 0)
            cv2.setTrackbarPos("Super Res  0-100", WIN, 0)
            cv2.setTrackbarPos("Color Trnsfr 0/1", WIN, 0)
            # phase 5
            cv2.setTrackbarPos("Text       0/1",   WIN, 0)
            cv2.setTrackbarPos("Text X %",         WIN, 10)
            cv2.setTrackbarPos("Text Y %",         WIN, 10)
            cv2.setTrackbarPos("Text Size",        WIN, 40)
            cv2.setTrackbarPos("Text R",           WIN, 255)
            cv2.setTrackbarPos("Text G",           WIN, 255)
            cv2.setTrackbarPos("Text B",           WIN, 255)
            cv2.setTrackbarPos("Qual Enh  0-100",  WIN, 0)
            cv2.setTrackbarPos("Compress  1-100",  WIN, 95)
            state["text_content"] = "Your Text Here"
            print("\nReset to defaults")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
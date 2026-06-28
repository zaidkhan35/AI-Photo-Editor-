
# AI Features: background removal, super resolution, denoise, auto enhance, color transfer
# Imports Phase 1 + 2 + 3 pipelines

import cv2
import numpy as np
from basic_editor     import original, h, w, state, show_preview, nothing
from artistic_effects import run_full_pipeline
from face_detection   import run_phase3_pipeline, face_cascade


state["bg_remove"]      = 0
state["super_res"]      = 0
state["denoise"]        = 0
state["auto_enhance"]   = 0
state["color_transfer"] = 0

color_ref_image = None


def apply_bg_remove(image):
    try:
        from rembg import remove
        from PIL   import Image

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output  = remove(pil_img)
        result  = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2BGRA)

        bg      = np.ones((result.shape[0], result.shape[1], 3),
                          dtype=np.uint8) * 255
        alpha   = result[:, :, 3] / 255.0

        for c in range(3):
            bg[:, :, c] = (alpha * result[:, :, c] +
                          (1 - alpha) * bg[:, :, c]).astype(np.uint8)
        return bg

    except ImportError:
        print("rembg not installed. Run: pip install \"rembg[cpu]\"")
        return image


def apply_denoise(image, strength):
    h_val = max(3, int(strength * 0.27))
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h                  = h_val,
        hColor             = h_val,
        templateWindowSize = 7,
        searchWindowSize   = 21
    )


def apply_auto_enhance(image, strength):
    clip    = 1.0 + (strength / 100.0) * 3.0
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_color_transfer(image, reference):
    if reference is None:
        return image

    src = cv2.cvtColor(image,     cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    for c in range(3):
        src_mean, src_std = src[:, :, c].mean(), src[:, :, c].std()
        ref_mean, ref_std = ref[:, :, c].mean(), ref[:, :, c].std()
        src[:, :, c] = (src[:, :, c] - src_mean) * \
                       (ref_std / (src_std + 1e-6)) + ref_mean

    src = np.clip(src, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src, cv2.COLOR_LAB2BGR)


def apply_super_resolution(image, strength):
    scale  = 1.0 + (strength / 100.0) * 3.0
    h_img, w_img = image.shape[:2]
    new_w  = int(w_img * scale)
    new_h  = int(h_img * scale)

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("ESPCN_x4.pb")
        sr.setModel("espcn", 4)
        result = sr.upsample(image)
        return cv2.resize(result, (new_w, new_h))

    except Exception:
        return cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_CUBIC)


def run_phase4_pipeline():
    result = run_phase3_pipeline()

    if state["denoise"]      > 0:
        result = apply_denoise(result,       state["denoise"])

    if state["auto_enhance"] > 0:
        result = apply_auto_enhance(result,  state["auto_enhance"])

    # fix — only run color transfer if reference image is loaded
    if state["color_transfer"] == 1 and color_ref_image is not None:
        result = apply_color_transfer(result, color_ref_image)

    if state["super_res"]    > 0:
        result = apply_super_resolution(result, state["super_res"])

    if state["bg_remove"]    == 1:
        result = apply_bg_remove(result)

    return result


if __name__ == "__main__":

    WIN = "AI Features  |  S=Save  R=Reset  C=Load Color Ref  Q=Quit"
    cv2.namedWindow(WIN)

    # phase 1 sliders
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

    # phase 2 sliders
    cv2.createTrackbar("Cartoon   0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch BW 0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sketch CL 0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Stylize   0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Edges     0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Emboss    0-100",  WIN, 0, 100, nothing)
    cv2.createTrackbar("Sepia     0-100",  WIN, 0, 100, nothing)

    # phase 3 sliders
    cv2.createTrackbar("Face Mode  0-4",   WIN, 0,  4,  nothing)
    cv2.createTrackbar("Min Neighbors",    WIN, 5,  20, nothing)
    cv2.createTrackbar("Scale x10",        WIN, 12, 20, nothing)

    # phase 4 sliders
    cv2.createTrackbar("BG Remove  0/1",   WIN, 0,  1,   nothing)
    cv2.createTrackbar("Denoise    0-100", WIN, 0,  100, nothing)
    cv2.createTrackbar("Enhance    0-100", WIN, 0,  100, nothing)
    cv2.createTrackbar("Super Res  0-100", WIN, 0,  100, nothing)
    cv2.createTrackbar("Color Trnsfr 0/1", WIN, 0,  1,   nothing)

    print("\n--- AI Photo Editor | Phase 4 ---")
    print("BG Remove    — removes background, white fill behind subject")
    print("Denoise      — removes grain and noise from photo")
    print("Enhance      — smart contrast enhancement with CLAHE")
    print("Super Res    — upscales image with better quality than resize")
    print("Color Trnsfr — press C first to load a reference image")
    print("")
    print("S=Save  |  R=Reset  |  C=Load color reference  |  Q=Quit")

    while True:

        # phase 1
        state["grayscale"]       = cv2.getTrackbarPos("Grayscale 0/1",    WIN)
        state["brightness"]      = cv2.getTrackbarPos("Brightness",       WIN)
        state["contrast"]        = cv2.getTrackbarPos("Contrast",         WIN)
        state["blur"]            = cv2.getTrackbarPos("Blur",             WIN)
        state["sharpen"]         = cv2.getTrackbarPos("Sharpen  0/1",     WIN)
        state["scale"]           = cv2.getTrackbarPos("Scale %",          WIN)
        state["rotate"]          = cv2.getTrackbarPos("Rotate",           WIN)
        state["flip"]            = cv2.getTrackbarPos("Flip  0-3",        WIN)
        state["crop_x"]          = cv2.getTrackbarPos("Crop X",           WIN)
        state["crop_y"]          = cv2.getTrackbarPos("Crop Y",           WIN)
        state["crop_w"]          = cv2.getTrackbarPos("Crop W",           WIN)
        state["crop_h"]          = cv2.getTrackbarPos("Crop H",           WIN)

        # phase 2
        state["cartoon"]         = cv2.getTrackbarPos("Cartoon   0-100",  WIN)
        state["sketch_bw"]       = cv2.getTrackbarPos("Sketch BW 0-100",  WIN)
        state["sketch_color"]    = cv2.getTrackbarPos("Sketch CL 0-100",  WIN)
        state["stylize"]         = cv2.getTrackbarPos("Stylize   0-100",  WIN)
        state["edges"]           = cv2.getTrackbarPos("Edges     0-100",  WIN)
        state["emboss"]          = cv2.getTrackbarPos("Emboss    0-100",  WIN)
        state["sepia"]           = cv2.getTrackbarPos("Sepia     0-100",  WIN)

        # phase 3
        state["face_mode"]       = cv2.getTrackbarPos("Face Mode  0-4",   WIN)
        state["min_neighbors"]   = cv2.getTrackbarPos("Min Neighbors",    WIN)
        state["scale_factor_x10"]= cv2.getTrackbarPos("Scale x10",        WIN)

        # phase 4
        state["bg_remove"]       = cv2.getTrackbarPos("BG Remove  0/1",   WIN)
        state["denoise"]         = cv2.getTrackbarPos("Denoise    0-100",  WIN)
        state["auto_enhance"]    = cv2.getTrackbarPos("Enhance    0-100",  WIN)
        state["super_res"]       = cv2.getTrackbarPos("Super Res  0-100",  WIN)
        state["color_transfer"]  = cv2.getTrackbarPos("Color Trnsfr 0/1",  WIN)

        result = run_phase4_pipeline()
        show_preview(result, WIN)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite("output_phase4.jpg", result)
            print("\nSaved as output_phase4.jpg")

        elif key == ord('c'):
            path = input("\nEnter path to reference image: ").strip()
            ref  = cv2.imread(path)
            if ref is not None:
                color_ref_image = ref
                print(f"Color reference loaded: {path}")
            else:
                print("Could not load that image.")

        elif key == ord('r'):
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
            cv2.setTrackbarPos("Cartoon   0-100",  WIN, 0)
            cv2.setTrackbarPos("Sketch BW 0-100",  WIN, 0)
            cv2.setTrackbarPos("Sketch CL 0-100",  WIN, 0)
            cv2.setTrackbarPos("Stylize   0-100",  WIN, 0)
            cv2.setTrackbarPos("Edges     0-100",  WIN, 0)
            cv2.setTrackbarPos("Emboss    0-100",  WIN, 0)
            cv2.setTrackbarPos("Sepia     0-100",  WIN, 0)
            cv2.setTrackbarPos("Face Mode  0-4",   WIN, 0)
            cv2.setTrackbarPos("Min Neighbors",    WIN, 5)
            cv2.setTrackbarPos("Scale x10",        WIN, 12)
            cv2.setTrackbarPos("BG Remove  0/1",   WIN, 0)
            cv2.setTrackbarPos("Denoise    0-100", WIN, 0)
            cv2.setTrackbarPos("Enhance    0-100", WIN, 0)
            cv2.setTrackbarPos("Super Res  0-100", WIN, 0)
            cv2.setTrackbarPos("Color Trnsfr 0/1", WIN, 0)
            color_ref_image = None
            print("\nReset to defaults")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
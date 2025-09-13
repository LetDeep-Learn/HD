# # make_lr_simple.py
# # Simple, old-style script to create degraded low-res images from a folder of HR images.
# # Edit the variables below, then run: python make_lr_simple.py

# import os
# import sys
# import random
# from PIL import Image, ImageFilter
# import numpy as np
# import cv2

# # ==========================
# # CONFIG - edit these values
# # ==========================
# INPUT_DIR = r"D:\Omkar Sonawale\data\models\HD\dataset\HR"         # folder containing original high-res images
# OUTPUT_DIR = r"D:\Omkar Sonawale\data\models\HD\dataset\LR"        # folder where degraded images will be saved (same filenames)
# DOWNSCALE = 4              # integer downscale factor, then upscale back (set 1 to skip downscaling)
# BLUR_SIGMA = 1.0           # Gaussian blur radius (0 to disable)
# JPEG_QUALITY = 40          # JPEG quality for recompression (1-95)
# NOISE_STD = 5.0            # gaussian noise std dev (0 to disable)
# DETERMINISTIC = False      # if True, results are reproducible
# SEED = 42                  # random seed (when deterministic True)
# RECURSIVE = False          # if True, walk subdirectories and preserve folder structure
# VERBOSE = True             # prints progress info
# # ==========================

# # Supported image extensions
# EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')

# def ensure_out_dir(out_dir):
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

# def list_images(input_dir, recursive=False):
#     files = []
#     if recursive:
#         for root, dirs, filenames in os.walk(input_dir):
#             for fn in filenames:
#                 if fn.lower().endswith(EXTS):
#                     files.append(os.path.join(root, fn))
#     else:
#         for fn in os.listdir(input_dir):
#             path = os.path.join(input_dir, fn)
#             if os.path.isfile(path) and fn.lower().endswith(EXTS):
#                 files.append(path)
#     return files

# def degrade_image(pil_img, downscale=4, blur_sigma=1.0, jpeg_quality=40, noise_std=5.0, rng=None):
#     # convert to RGB
#     img = pil_img.convert('RGB')
#     w, h = img.size

#     # downscale & upsample (bicubic)
#     if downscale is None:
#         downscale = 1
#     if downscale > 1:
#         dw = max(1, int(w / downscale))
#         dh = max(1, int(h / downscale))
#         img = img.resize((dw, dh), resample=Image.BICUBIC)
#         img = img.resize((w, h), resample=Image.BICUBIC)

#     # blur
#     if blur_sigma and blur_sigma > 0:
#         img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

#     # to numpy RGB
#     arr = np.array(img).astype(np.uint8)

#     # JPEG compress via OpenCV encode/decode
#     try:
#         q = int(max(1, min(95, jpeg_quality)))
#         ok, enc = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
#         if ok:
#             dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
#             arr = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
#     except Exception:
#         # if opencv fails for some reason, continue without compression
#         pass

#     # add gaussian noise
#     if noise_std and noise_std > 0:
#         if rng is None:
#             noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
#         else:
#             # use provided rng for deterministic behavior
#             noise = rng.normal(0, noise_std, arr.shape).astype(np.float32)
#         arr = arr.astype(np.float32) + noise
#         arr = np.clip(arr, 0, 255).astype(np.uint8)

#     return Image.fromarray(arr)

# def save_image_same_name(src_path, dst_root, input_root, img_pil, jpeg_quality=40):
#     # preserve relative path if recursive, else just same filename in dst_root
#     if RECURSIVE:
#         rel = os.path.relpath(src_path, input_root)
#         dst_path = os.path.join(dst_root, rel)
#         dst_dir = os.path.dirname(dst_path)
#         if not os.path.exists(dst_dir):
#             os.makedirs(dst_dir)
#     else:
#         fname = os.path.basename(src_path)
#         dst_path = os.path.join(dst_root, fname)

#     # save respecting original extension (for JPEG, set quality)
#     ext = os.path.splitext(dst_path)[1].lower()
#     try:
#         if ext in ('.jpg', '.jpeg'):
#             img_pil.save(dst_path, quality=jpeg_quality, subsampling=0)
#         else:
#             img_pil.save(dst_path)
#     except Exception:
#         # fallback save
#         img_pil.save(dst_path)

#     return dst_path

# def main():
#     if DETERMINISTIC:
#         np.random.seed(SEED)
#         random.seed(SEED)
#         rng = np.random.RandomState(SEED)
#     else:
#         rng = None

#     if not os.path.exists(INPUT_DIR):
#         print("Input folder does not exist:", INPUT_DIR)
#         sys.exit(1)

#     ensure_out_dir(OUTPUT_DIR)
#     all_files = list_images(INPUT_DIR, recursive=RECURSIVE)
#     n = len(all_files)
#     if n == 0:
#         print("No images found in", INPUT_DIR)
#         sys.exit(1)

#     print("Processing {} images...".format(n))
#     count = 0
#     for path in all_files:
#         try:
#             pil = Image.open(path)
#         except Exception:
#             print("Skipped (cannot open):", path)
#             continue
#         degraded = degrade_image(pil,
#                                  downscale=DOWNSCALE,
#                                  blur_sigma=BLUR_SIGMA,
#                                  jpeg_quality=JPEG_QUALITY,
#                                  noise_std=NOISE_STD,
#                                  rng=rng)
#         out_path = save_image_same_name(path, OUTPUT_DIR, INPUT_DIR, degraded, jpeg_quality=JPEG_QUALITY)
#         count += 1
#         if VERBOSE:
#             print("[{}/{}] -> {}".format(count, n, out_path))

#     print("Done. {} images processed.".format(count))

# if __name__ == "__main__":
#     main()


# make_lr_simple.py
# Simple, old-style script to create degraded low-res images from a folder of HR images.
# Edit the variables below, then run: python make_lr_simple.py

import os
import sys
import random
from PIL import Image, ImageFilter
import numpy as np
import cv2
from pathlib import Path
import traceback

# ==========================
# CONFIG - edit these values
# ==========================
INPUT_DIR = r"D:\Omkar Sonawale\data\models\HD\dataset\HR"         # folder containing original high-res images
OUTPUT_DIR = r"D:\Omkar Sonawale\data\models\HD\dataset\LR"        # folder where degraded images will be saved (same filenames)
DOWNSCALE = 4              # integer downscale factor, then upscale back (set 1 to skip downscaling)
BLUR_SIGMA = 1.0           # Gaussian blur radius (0 to disable)
JPEG_QUALITY = 50         # JPEG quality for recompression (1-95)
NOISE_STD = 5.0            # gaussian noise std dev (0 to disable)
DETERMINISTIC = False      # if True, results are reproducible
SEED = 42                  # random seed (when deterministic True)
RECURSIVE = False          # if True, walk subdirectories and preserve folder structure
VERBOSE = True             # prints progress info
USE_TQDM = False           # try to show tqdm progress if you like (set True if tqdm installed)
# ==========================

# Supported image extensions
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')

# Try to import tqdm if the user wants it
if USE_TQDM:
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
        USE_TQDM = False
else:
    tqdm = None

def ensure_out_dir(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def list_images(input_dir, recursive=False):
    input_dir = Path(input_dir)
    files = []
    if recursive:
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                files.append(str(p))
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in EXTS:
                files.append(str(p))
    return files

def degrade_image(pil_img, downscale=4, blur_sigma=1.0, jpeg_quality=40, noise_std=5.0, rng=None):
    """
    Deterministic-friendly degradation:
     - downscale/upscale (bicubic)
     - optional gaussian blur
     - JPEG recompression via OpenCV
     - additive gaussian noise
    Returns a PIL.Image (RGB)
    """
    # convert to RGB (drops alpha)
    img = pil_img.convert('RGB')
    w, h = img.size

    # downscale & upsample (bicubic)
    if downscale is None:
        downscale = 1
    if downscale > 1:
        dw = max(1, int(w / downscale))
        dh = max(1, int(h / downscale))
        img = img.resize((dw, dh), resample=Image.BICUBIC)
        img = img.resize((w, h), resample=Image.BICUBIC)

    # blur
    if blur_sigma and blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    # to numpy RGB
    arr = np.array(img).astype(np.uint8)

    # JPEG compress via OpenCV encode/decode (best-effort)
    try:
        q = int(max(1, min(95, jpeg_quality)))
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                arr = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            else:
                if VERBOSE:
                    print("Warning: cv2.imdecode returned None for an image; skipping recompression for this file.")
        else:
            if VERBOSE:
                print("Warning: cv2.imencode returned False; skipping recompression for this file.")
    except Exception as e:
        if VERBOSE:
            print("Warning: OpenCV encode/decode failed, continuing without JPEG recompression. Error:", str(e))

    # add gaussian noise
    if noise_std and noise_std > 0:
        try:
            if rng is None:
                noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
            else:
                noise = rng.normal(0, noise_std, arr.shape).astype(np.float32)
            arr = arr.astype(np.float32) + noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception as e:
            if VERBOSE:
                print("Warning: adding noise failed for this image:", e)

    return Image.fromarray(arr)
def save_image_same_name(src_path, dst_root, input_root, img_pil, jpeg_quality=40):
    fname = os.path.splitext(os.path.basename(src_path))[0] + ".jpeg"
    dst_path = os.path.join(dst_root, fname)
    img_pil = img_pil.convert("RGB")  # ensure no alpha channel
    img_pil.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    return dst_path


# def save_image_same_name(src_path, dst_root, input_root, img_pil, jpeg_quality=40):
#     """
#     Save img_pil with the same filename to dst_root (preserve subfolders when RECURSIVE=True)
#     Returns the destination path (string). Exceptions are raised to caller.
#     """
#     src_path = str(src_path)
#     if RECURSIVE:
#         rel = os.path.relpath(src_path, input_root)
#         dst_path = os.path.join(dst_root, rel)
#         dst_dir = os.path.dirname(dst_path)
#         if not os.path.exists(dst_dir):
#             os.makedirs(dst_dir, exist_ok=True)
#     else:
#         fname = os.path.basename(src_path)
#         dst_path = os.path.join(dst_root, fname)

#     # save respecting original extension (for JPEG, set quality)
#     ext = os.path.splitext(dst_path)[1].lower()
#     try:
#         if ext in ('.jpg', '.jpeg'):
#             # Use optimize to get slightly smaller files, and avoid subsampling param issues across PIL versions
#             img_pil.save(dst_path, quality=jpeg_quality, optimize=True)
#         else:
#             img_pil.save(dst_path)
#     except Exception as e:
#         # try fallback: attempt saving with default options and print the error
#         try:
#             img_pil.save(dst_path)
#         except Exception as e2:
#             # re-raise a helpful message
#             raise RuntimeError(f"Failed to save image to {dst_path}. First error: {e}. Fallback error: {e2}")
#     return dst_path

def main():
    # Setup deterministic RNG if requested
    if DETERMINISTIC:
        np.random.seed(SEED)
        random.seed(SEED)
        rng = np.random.RandomState(SEED)
    else:
        rng = None

    # Basic sanity checks
    if os.path.abspath(INPUT_DIR) == os.path.abspath(OUTPUT_DIR):
        print("Warning: INPUT_DIR is the same as OUTPUT_DIR. This will overwrite your originals. Abort if that's not intended.")
        # continue, but user should change manually

    if not os.path.exists(INPUT_DIR):
        print("Input folder does not exist:", INPUT_DIR)
        sys.exit(1)

    ensure_out_dir(OUTPUT_DIR)
    all_files = list_images(INPUT_DIR, recursive=RECURSIVE)
    n = len(all_files)
    if n == 0:
        print("No images found in", INPUT_DIR)
        sys.exit(1)

    # Counters
    skipped = 0
    saved = 0
    failed = 0

    print("Processing {} images...".format(n))
    iterator = all_files
    if USE_TQDM and tqdm is not None:
        iterator = tqdm(all_files)

    for path in iterator:
        try:
            # use context manager to ensure file handles are closed promptly
            with Image.open(path) as pil:
                degraded = degrade_image(pil,
                                         downscale=DOWNSCALE,
                                         blur_sigma=BLUR_SIGMA,
                                         jpeg_quality=JPEG_QUALITY,
                                         noise_std=NOISE_STD,
                                         rng=rng)
                try:
                    out_path = save_image_same_name(path, OUTPUT_DIR, INPUT_DIR, degraded, jpeg_quality=JPEG_QUALITY)
                    saved += 1
                    if VERBOSE:
                        print("[{}/{}] Saved -> {}".format(saved + skipped + failed, n, out_path))
                except Exception as e_save:
                    failed += 1
                    print(f"[{saved + skipped + failed}/{n}] Failed to save {path}. Error: {e_save}")
                    if VERBOSE:
                        traceback.print_exc()
        except Exception as e_open:
            skipped += 1
            print(f"[{saved + skipped + failed}/{n}] Skipped (cannot open): {path}. Error: {e_open}")
            if VERBOSE:
                traceback.print_exc()

    # Summary
    print("=" * 60)
    print(f"Done. Total files found: {n}")
    print(f"Saved: {saved}")
    print(f"Skipped (open failures): {skipped}")
    print(f"Failed (save or processing errors): {failed}")
    print("=" * 60)

if __name__ == "__main__":
    main()


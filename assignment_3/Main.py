import cv2
import numpy as np
from pathlib import Path
import subprocess

# paths to pictures. for simplicity i just get them in the download folder. If i put them in the "correct" folder they will not load properly
LAMBO_PATH = r"C:\Users\erlin\Downloads\lambo.png"
SHAPES_PATH = r"C:\Users\erlin\Downloads\shapes-1.png"
TEMPLATE_PATH = r"C:\Users\erlin\Downloads\shapes_template.jpg"

# Output folder
OUTPUT_DIR = Path(r"C:\Users\erlin\Skole høstsemester 2025\IKT213_Abrahamsen\assignment_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------


def save_image(image, path: Path):
    path = str(path)
    ext = Path(path).suffix or ".png"
    try:
        # Encode to memory, then write to file (Unicode-safe)
        result, encimg = cv2.imencode(ext, image)
        if result:
            encimg.tofile(path)
            print(f"✅ Saved: {path}")
        else:
            print(f"❌ Failed to encode image for: {path}")
    except Exception as e:
        print(f"❌ Failed to save: {path} | Exception: {e}")


def sobel_edge_detection(image_path: str, save_path: Path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    mag_norm = np.uint8(255 * (magnitude / (magnitude.max() + 1e-10)))

    save_image(mag_norm, save_path)


def canny_edge_detection(image_path: str, threshold_1: int, threshold_2: int, save_path: Path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    save_image(edges, save_path)


def template_match(image_path: str, template_path: str, save_path: Path, threshold: float = 0.9):
    img = cv2.imread(image_path)
    tpl = cv2.imread(template_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    if tpl is None:
        raise FileNotFoundError(f"Could not open template: {template_path}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = tpl_gray.shape[:2]
    out = img.copy()
    matched_any = False
    for pt in zip(*loc[::-1]):
        matched_any = True
        cv2.rectangle(out, pt, (pt[0]+w, pt[1]+h), color=(0,0,255), thickness=2)
    if not matched_any:
        print(f"No matches found above threshold={threshold}")
    save_image(out, save_path)


def resize(image_path: str, scale_factor: int, up_or_down: str, save_path: Path):
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    out = img.copy()
    if up_or_down.lower() == "up":
        for _ in range(scale_factor):
            out = cv2.pyrUp(out)
    elif up_or_down.lower() == "down":
        for _ in range(scale_factor):
            out = cv2.pyrDown(out)
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    save_image(out, save_path)


# Main
if __name__ == "__main__":
    sobel_out = OUTPUT_DIR / "lambo_sobel.png"
    canny_out = OUTPUT_DIR / "lambo_canny.png"
    tpl_out = OUTPUT_DIR / "shapes_template_matched.png"
    resize_up_out = OUTPUT_DIR / "lambo_pyrup_x2.png"
    resize_down_out = OUTPUT_DIR / "lambo_pyrdown_x2.png"

    try:
        sobel_edge_detection(LAMBO_PATH, sobel_out)
        canny_edge_detection(LAMBO_PATH, 50, 50, canny_out)
        template_match(SHAPES_PATH, TEMPLATE_PATH, tpl_out)
        resize(LAMBO_PATH, 2, "up", resize_up_out)
        resize(LAMBO_PATH, 2, "down", resize_down_out)
    except Exception as e:
        print("Error:", e)

    print("\nAll operations attempted. Output folder:", OUTPUT_DIR)

    # Open output folder.
    try:
        subprocess.Popen(f'explorer "{OUTPUT_DIR}"')
    except Exception as e:
        print("Could not open folder automatically:", e)

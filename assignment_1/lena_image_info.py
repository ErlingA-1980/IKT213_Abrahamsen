import os
import numpy as np
import cv2

    #hent
def load_image_unicode_safe(path, flags=cv2.IMREAD_COLOR):

    try:
        data = np.fromfile(path, dtype=np.uint8)  # handles Unicode paths
        if data.size > 0:
            img = cv2.imdecode(data, flags)
            return img
    except Exception:
        pass
    return None

    #print
def print_image_information(image):
    """Prints height, width, channels, size, and data type of the image."""
    if image is None:
        print("Error: Image is None (failed to load).")
        return

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size = image.size
    dtype = image.dtype

    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {size}")
    print(f"Data type: {dtype}")

def main():
    img_path = r"C:\Users\erlin\Skole høstsemester 2025\Ikt213g24h\assignments\solutions\Assignment_1\lena-1.png"

    if not os.path.exists(img_path):
        print(f"File not found at:\n  {img_path}")
        return

    # modifisering for å unngå feilkode(unicode safe)
    img = load_image_unicode_safe(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Failed to load image from:\n  {img_path}")
        return

    # Print bildeinfo
    print_image_information(img)

    # vis bilde (så ser en om infoen stemmer)
    try:
        cv2.imshow("Lena Image", img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC
            cv2.destroyAllWindows()
        elif k == ord('s'):
            out_name = os.path.join(os.path.dirname(img_path), "lena_copy.png")
            cv2.imwrite(out_name, img)
            print(f"Image saved as {out_name}")
            cv2.destroyAllWindows()
    except cv2.error:
        print("GUI not available; skipped cv2.imshow().")

if __name__ == "__main__":
    main()

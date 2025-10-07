import cv2
import os
import numpy as np

# Load image
img_path = "C:/Users/erlin/OneDrive/Bilder/Skole/lena.png"
#had to place the picture another place to be able to read AND understand the information in it. dont understand why"

print("Final path:", img_path)
print("Exists:", os.path.exists(img_path)) #also just some residual after arguing with the code to see if it found the pic

image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image from path: {img_path}. "
        "Check if the file exists, the path is correct, "
        "and the file is a valid image format."
    )
#some residual coding from before i just moved the picture, left it for you to see some of the process.

# 1. Padding
def padding(image, border_width):
    padded_img = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT
    )
    cv2.imwrite("padded.png", padded_img)
    return padded_img


# 2. Cropping
def crop(image, x_0, x_1, y_0, y_1):
    cropped_img = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("cropped.png", cropped_img)
    return cropped_img


# 3. Resize
def resize(image, width, height):
    resized_img = cv2.resize(image, (width, height))
    cv2.imwrite("resized.png", resized_img)
    return resized_img


# 4. Manual Copy
def copy(image, emptyPictureArray):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("copied.png", emptyPictureArray)
    return emptyPictureArray


# 5. Grayscale
def grayscale(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale.png", gray_img)
    return gray_img


# 6. HSV Conversion
def hsv(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png", hsv_img)
    return hsv_img


# 7. Hue Shift vectorized/fixed
def hue_shifted(image, hue):
    shifted = (image.astype(np.uint16) + hue) % 256
    shifted = shifted.astype(np.uint8)
    cv2.imwrite("hue_shifted.png", shifted)
    return shifted


# 8. Smoothing
def smoothing(image):
    smooth_img = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite("smoothed.png", smooth_img)
    return smooth_img


# 9. Rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("Rotation angle must be 90 or 180")
    cv2.imwrite(f"rotated_{rotation_angle}.png", rotated_img)
    return rotated_img


# Testing testing, think it does all it should :-)
if __name__ == "__main__":
    padded = padding(image, 100)
    cropped = crop(image, 80, image.shape[1] - 130, 80, image.shape[0] - 130)
    resized = resize(image, 200, 200)

    h, w, c = image.shape
    empty_array = np.zeros((h, w, 3), dtype=np.uint8)
    copied = copy(image, empty_array)

    gray = grayscale(image)
    hsv_img = hsv(image)

    hue_shift = hue_shifted(image, 50)

    smooth = smoothing(image)
    rotated_90 = rotation(image, 90)
    rotated_180 = rotation(image, 180)

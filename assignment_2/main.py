import cv2
import numpy as np


def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/Part_1.png", reflect)

def crop(image, x_0, x_1,  y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("solutions/Part_2.png", cropped_image)

def resize(image, width, height):
    reSized = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/Part_3.png", reSized)

def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y,x,c] = image[y,x,c]
    cv2.imwrite("solutions/Part_4.png", emptyPictureArray)

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/Part_5.png", gray)

def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/Part_6.png", hsvImage)

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                colorValue = image[y,x,c] + hue
                if colorValue > 255:
                    colorValue = 255
                elif colorValue < 0:
                    colorValue = 0
                emptyPictureArray[y,x,c] = colorValue
    cv2.imwrite("solutions/Part_7.png", emptyPictureArray)

def smoothing(image):
    dst = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imwrite("solutions/Part_8.png", dst)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif rotation_angle == 180:
        rotate = cv2.ROTATE_180
    else:
        return

    rotated = cv2.rotate(image, rotate)
    cv2.imwrite("solutions/Part_9.png", rotated)

def main():
    image = cv2.imread("images/lena-2.png", cv2.IMREAD_UNCHANGED)
    padding(image, 100)
    height, width, channels = image.shape
    crop(image, 80, width-130 ,80,height-130)
    resize(image, 200,200)

    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, emptyPictureArray)

    grayscale(image)
    hsv(image)
    hue_shifted(image, emptyPictureArray, 50)
    smoothing(image)
    rotation(image, 180)

if __name__ == '__main__':
    main()
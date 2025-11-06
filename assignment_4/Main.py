import cv2
import numpy as np
import os


# 1. Harris Corner Detection
def harris_corner_detection(reference_image_path, save_path="harris.png"):

    print("[INFO] Starting Harris Corner Detection...")

    # Read the image
    img = cv2.imread(reference_image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {reference_image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris corner detection
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # Thrshold and corners
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Save
    cv2.imwrite(save_path, img)
    print(f"[INFO] Harris corners saved as: {os.path.abspath(save_path)}")



# 2. Feature-Based Image Alignment (SIFT)

def align_images_sift(reference_image_path, image_to_align_path, save_path="aligned_image.png"):

    print("[INFO] Starting SIFT feature-based alignment...")

    # Load both images
    ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    img_to_align = cv2.imread(image_to_align_path, cv2.IMREAD_GRAYSCALE)

    # If second image doesn't exist, just explain and exit
    if img_to_align is None:
        print("[INFO] No second image provided — function implemented but not executed.")
        print("       This function is ready for use when a second image is available.")
        return

    # Initialize SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(img_to_align, None)

    # Match features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using Lowe’s ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"[INFO] Found {len(good_matches)} good matches.")

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = ref_img.shape
        aligned = cv2.warpPerspective(cv2.imread(image_to_align_path), H, (w, h))
        cv2.imwrite(save_path, aligned)
        print(f"[INFO] Aligned image saved as: {os.path.abspath(save_path)}")
    else:
        print("[WARN] Not enough matches to align the images.")



# 3. Run main logic

if __name__ == "__main__":
    base_path = r"C:\Users\erlin\Skole_hostsemester_2025\IKT213_Abrahamsen\assignment_4"
    reference_img_path = os.path.join(base_path, "reference_img.png")

    # Run Harris corner detection
    harris_corner_detection(reference_img_path, os.path.join(base_path, "harris.png"))

    # Demonstrate feature-based alignment function (SIFT)
    # Not required to actually run if only one image is available
    align_images_sift(reference_img_path, reference_img_path)

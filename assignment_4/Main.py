import cv2
import numpy as np
import os

# ===============================
# Paths
# ===============================
base_path = r"C:\Users\erlin\Skole_hostsemester_2025\IKT213_Abrahamsen\assignment_4"

reference_path = os.path.join(base_path, "reference_img.png")
align_path = os.path.join(base_path, "align_this.jpg")

harris_output = os.path.join(base_path, "harris.png")
matches_output = os.path.join(base_path, "matches.jpg")
aligned_output = os.path.join(base_path, "aligned.jpg")


# ===============================
# 1. Harris Corner Detection
# ===============================
def detect_harris(reference_img, save_path):
    gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    corner_img = reference_img.copy()
    threshold = 0.1 * dst.max()
    corners = np.where(dst > threshold)
    corner_img[corners] = [0, 0, 255]

    cv2.imwrite(save_path, corner_img)
    print(f"[INFO] Harris corners saved to {save_path}")


# ===============================
# 2. SIFT Alignment + Matches
# ===============================
def align_images_sift_custom(img_to_align, ref_img, max_features=5000, good_match_percent=0.7,
                             save_matches_path=matches_output, save_aligned_path=aligned_output):

    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_features)
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_align, des_align = sift.detectAndCompute(gray_align, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des_ref, des_align, k=2)

    # Lowe's ratio test
    good_matches = [m1 for m1, m2 in matches if m1.distance < good_match_percent * m2.distance]

    # Draw matches
    match_img = cv2.drawMatches(ref_img, kp_ref, img_to_align, kp_align, good_matches, None, flags=2)
    cv2.imwrite(save_matches_path, match_img)
    print(f"[INFO] Matches saved to {save_matches_path}")

    if len(good_matches) < 4:
        print("[WARN] Not enough matches for homography")
        return

    # Homography
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_align[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[ERROR] Homography could not be computed")
        return

    h, w, _ = ref_img.shape
    aligned_img = cv2.warpPerspective(img_to_align, H, (w, h))
    cv2.imwrite(save_aligned_path, aligned_img)
    print(f"[INFO] Aligned image saved to {save_aligned_path}")


# ===============================
# 3. Main Execution
# ===============================
if __name__ == "__main__":

    reference_img = cv2.imread(reference_path)
    img_to_align = cv2.imread(align_path)

    if reference_img is None or img_to_align is None:
        print("[ERROR] Could not load images. Check paths and file integrity.")
    else:
        # Harris corners
        detect_harris(reference_img, harris_output)

        # SIFT alignment + matches
        align_images_sift_custom(img_to_align, reference_img, max_features=5000)

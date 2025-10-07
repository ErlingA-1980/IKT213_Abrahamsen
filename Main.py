import cv2
import numpy as np #from earlier version of the program#
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


def preprocess_fingerprint(image_path):
    """Convert fingerprint image to binary form, should be easier feature extraction."""
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


def match_fingerprints(img1_path, img2_path, method="sift_flann"):
    """Compare two fingerprint images using SIFT+FLANN or ORB+BF, combined both versions in one program for simplicity"""
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    if method == "sift_flann":
        detector = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0, None

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    elif method == "orb_bf":
        detector = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0, None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    else:
        raise ValueError("Unknown method. Choose 'sift_flann' or 'orb_bf'.")

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return len(good_matches), match_img
#works...#

def process_dataset(dataset_path, results_folder, method="sift_flann", threshold=20, draw_matches=True):
    """Evaluate fingerprint pairs and compute confusion matrix."""
    y_true, y_pred = [], []
    os.makedirs(results_folder, exist_ok=True)

    total_time = 0
    folders = sorted(os.listdir(dataset_path))

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        if len(images) != 2:
            print(f"Skipping {folder}: expected 2 images, found {len(images)}")
            continue

        img1 = os.path.join(folder_path, images[0])
        img2 = os.path.join(folder_path, images[1])

        start = time.time()
        match_count, match_img = match_fingerprints(img1, img2, method=method)
        end = time.time()
        total_time += (end - start)

        actual_match = 1 if "same" in folder.lower() else 0
        predicted_match = 1 if match_count > threshold else 0

        y_true.append(actual_match)
        y_pred.append(predicted_match)

        result_label = "MATCH" if predicted_match else "NO_MATCH"
        print(f"{folder}: {result_label} ({match_count} good matches)")

        if draw_matches and match_img is not None:
            output_file = os.path.join(results_folder, f"{folder}_{method}_{result_label}.png")
            cv2.imwrite(output_file, match_img)

    avg_time = total_time / len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Different (0)", "Same (1)"]

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({method})")
    plt.show()

    print(f"\nAverage time per comparison ({method}): {avg_time:.3f} seconds\n")
    return avg_time, cm



# Just to make it run easily without command prompt

if __name__ == "__main__":
    dataset_path = r"C:\Users\erlin\Skole_hostsemester_2025\IKT213_Abrahamsen\Fingerprint\datasets"
    results_folder = r"C:\Users\erlin\Skole_hostsemester_2025\IKT213_Abrahamsen\Fingerprint\results"

    print("Running SIFT + FLANN pipeline...")
    sift_time, sift_cm = process_dataset(dataset_path, os.path.join(results_folder, "sift_flann"), method="sift_flann", threshold=20)

    print("\nRunning ORB + BF pipeline...")
    orb_time, orb_cm = process_dataset(dataset_path, os.path.join(results_folder, "orb_bf"), method="orb_bf", threshold=15)

    print("\n--- Summary ---")
    print(f"SIFT + FLANN → avg time: {sift_time:.3f}s")
    print(f"ORB  +  BF   → avg time: {orb_time:.3f}s")

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import time
import psutil


def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


def orb_match_fingerprints(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None  # Return 0 matches if no descriptors found

    # Use Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN Match
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Draw only good matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img


def orb_process_dataset(dataset_path, results_folder):
    start = time.perf_counter()
    process = psutil.Process(os.getpid())
    cpu_start = process.cpu_times()

    threshold = 20  # Adjust this based on tests
    y_true = []  # True labels (1 for same, 0 for different)
    y_pred = []  # Predicted labels

    # Create results folder if it does not exist
    os.makedirs(results_folder, exist_ok=True)

    # Loop through all subdirectories
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):  # Check if it's a valid directory
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue  # Skip if the folder doesn't have exactly 2 images
            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])
            match_count, match_img = orb_match_fingerprints(img1_path, img2_path)

            # Determine the ground truth (expected label)
            actual_match = 1 if "same" in folder.lower() else 0  # 1 for same, 0 for different
            y_true.append(actual_match)

            # Decision based on good matches count
            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)
            result = "orb_bf_matched" if predicted_match == 1 else "orb_bf_unmatched"
            print(f"{folder}: {result.upper()} ({match_count} good matches)")

            # Save match image in the results folder
            if match_img is not None:
                match_img_filename = f"{folder}_{result}.png"
                match_img_path = os.path.join(results_folder, match_img_filename)
                cv2.imwrite(match_img_path, match_img)
                print(f"Saved match image at: {match_img_path}")

    # Compute and display confusion matrix
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix orb_bf")
    end = time.perf_counter()
    cpu_end = process.cpu_times()
    print("Elapsed:", end - start, "seconds")
    print("Memory (MB):", process.memory_info().rss / 1024 ** 2)
    print("User CPU time:", cpu_end.user - cpu_start.user, "s")
    plt.show()

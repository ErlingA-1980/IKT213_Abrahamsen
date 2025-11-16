import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import psutil

# ------------------------- Core Functions -------------------------

def preprocess_fingerprint(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

def load_general_image(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    return img

def match_images(img1_path, img2_path, method="sift_flann", fingerprint=True):
    if fingerprint:
        img1 = preprocess_fingerprint(img1_path)
        img2 = preprocess_fingerprint(img2_path)
    else:
        img1 = load_general_image(img1_path)
        img2 = load_general_image(img2_path)

    if method == "sift_flann":
        detector = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return 0, None
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
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
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return len(good_matches), match_img

# ------------------- Evaluate Folder Dataset -------------------

def evaluate_dataset(dataset_path, results_folder, method="sift_flann", threshold=20):
    y_true, y_pred = [], []
    os.makedirs(results_folder, exist_ok=True)

    start_time = time.perf_counter()
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / 1024 ** 2

    folders = sorted(os.listdir(dataset_path))
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        if len(images) != 2:
            print(f"Skipping {folder}: expected 2 images, found {len(images)}")
            continue

        img1_path = os.path.join(folder_path, images[0])
        img2_path = os.path.join(folder_path, images[1])

        match_count, match_img = match_images(img1_path, img2_path, method=method, fingerprint=True)

        actual_match = 1 if "same" in folder.lower() else 0
        predicted_match = 1 if match_count > threshold else 0

        y_true.append(actual_match)
        y_pred.append(predicted_match)

        result_label = "MATCH" if predicted_match else "NO_MATCH"
        print(f"{folder}: {result_label} ({match_count} good matches)")

        if match_img is not None:
            save_path = os.path.join(results_folder, f"{folder}_{method}_{result_label}.png")
            cv2.imwrite(save_path, match_img)

    end_time = time.perf_counter()
    mem_end = process.memory_info().rss / 1024 ** 2

    acc = sum(np.array(y_true) == np.array(y_pred)) / len(y_true) if y_true else 0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # force both classes

    if y_true:
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different (0)", "Same (1)"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix ({method})")
        plt.show(block=False)
        plt.pause(0.1)

    return end_time - start_time, mem_end - mem_start, acc, cm

# ------------------- Evaluate UiA Images -------------------

def evaluate_uia_images(dataset_path, results_folder, method="sift_flann", threshold=20):
    # Automatically find exactly two image files
    img_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(img_files) != 2:
        raise ValueError(f"UiA dataset folder must contain exactly 2 images, found {len(img_files)}")

    img_paths = [os.path.join(dataset_path, f) for f in img_files]

    y_true, y_pred = [], []
    os.makedirs(results_folder, exist_ok=True)

    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    mem_start = process.memory_info().rss / 1024 ** 2

    match_count, match_img = match_images(img_paths[0], img_paths[1], method=method, fingerprint=False)

    actual_match = 1  # assume same person
    predicted_match = 1 if match_count > threshold else 0

    y_true.append(actual_match)
    y_pred.append(predicted_match)

    result_label = "MATCH" if predicted_match else "NO_MATCH"
    print(f"{os.path.basename(img_paths[0])} vs {os.path.basename(img_paths[1])}: {result_label} ({match_count} good matches)")

    if match_img is not None:
        save_path = os.path.join(results_folder, f"{os.path.basename(img_paths[0])}_{os.path.basename(img_paths[1])}_{method}_{result_label}.png")
        cv2.imwrite(save_path, match_img)

    end_time = time.perf_counter()
    mem_end = process.memory_info().rss / 1024 ** 2

    acc = sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # force both classes

    plt.figure(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different (0)", "Same (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({method})")
    plt.show(block=False)
    plt.pause(0.1)

    return end_time - start_time, mem_end - mem_start, acc, cm

# ------------------- Main Comparison -------------------

def run_comparison():
    datasets = {
        "Fingerprints": r"C:\Users\erlin\Downloads\data_check",
        "UiA": r"C:\Users\erlin\Downloads\uia"
    }

    results_root = r"C:\Users\erlin\Skole_hostsemester_2025\IKT213_Abrahamsen\Fingerprint\results"
    os.makedirs(results_root, exist_ok=True)

    methods = ["sift_flann", "orb_bf"]
    threshold_dict = {"sift_flann": 20, "orb_bf": 15}

    summary = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\n=== Evaluating dataset: {dataset_name} ===")
        summary[dataset_name] = {}

        if dataset_name == "UiA":
            for method in methods:
                print(f"\n--- Method: {method} (UiA) ---")
                results_folder = os.path.join(results_root, f"UiA_{method}")
                time_taken, mem_used, acc, cm = evaluate_uia_images(dataset_path, results_folder, method=method, threshold=threshold_dict[method])
                summary[dataset_name][method] = {"time_s": time_taken, "mem_MB": mem_used, "accuracy": acc, "confusion_matrix": cm}

        else:
            for method in methods:
                print(f"\n--- Method: {method} ---")
                results_folder = os.path.join(results_root, f"{dataset_name}_{method}")
                time_taken, mem_used, acc, cm = evaluate_dataset(dataset_path, results_folder, method=method, threshold=threshold_dict[method])
                summary[dataset_name][method] = {"time_s": time_taken, "mem_MB": mem_used, "accuracy": acc, "confusion_matrix": cm}

    # ------------------- Summary -------------------
    print("\n=== SUMMARY COMPARISON ===")
    for dataset_name in datasets.keys():
        print(f"\nDataset: {dataset_name}")
        for method in methods:
            stats = summary[dataset_name][method]
            print(f"{method}: time={stats['time_s']:.2f}s, mem={stats['mem_MB']:.2f}MB, accuracy={stats['accuracy']*100:.2f}%")

        sift = summary[dataset_name]["sift_flann"]
        orb = summary[dataset_name]["orb_bf"]
        if orb["time_s"] < sift["time_s"] and orb["mem_MB"] < sift["mem_MB"] and orb["accuracy"] >= sift["accuracy"]:
            print("=> ORB pipeline is more efficient overall.")
        else:
            print("=> SIFT pipeline is more accurate or ORB is faster but less accurate.")

# ------------------- Run Script -------------------
if __name__ == "__main__":
    run_comparison()

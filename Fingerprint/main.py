import os

import cv2

from ORB import orb_process_dataset, preprocess_fingerprint, orb_match_fingerprints
from SIFT import sift_process_dataset

fingerprint_dataset_path = r"./Data_check"
fingerprint_orb_results_folder = r"./orb_bf_"
fingerprint_sift_results_folder = r"./sift_flann_"

uia_dataset_path = r"./uia"
uia_orb_results_folder = r"./uia_orb_bf_"
uia_sift_results_folder = r"./uia_sift_flann_"

threshold = 20

img1_path = r"./uia/UiA front1.png"
img2_path = r"./uia/UiA front3.jpg"

match_count, match_img = orb_match_fingerprints(img1_path, img2_path)

predicted_match = 1 if match_count > threshold else 0

result = "orb_bf_matched" if predicted_match == 1 else "orb_bf_unmatched"
print(f"{result.upper()} ({match_count} good matches)")

match_img_path = os.path.join(uia_orb_results_folder, r"result.png")
cv2.imwrite(match_img_path, match_img)
print(f"Saved match image at: {match_img_path}")
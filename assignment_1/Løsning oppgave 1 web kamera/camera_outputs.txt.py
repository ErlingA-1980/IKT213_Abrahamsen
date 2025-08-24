import cv2
import os

def save_camera_info(output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the default camera (0 = first camera)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return

    # Get properties
    fps = cam.get(cv2.CAP_PROP_FPS)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Path for saving the text file
    output_file = os.path.join(output_dir, "camera_outputs.txt")

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"fps: {fps:.2f}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print(f"Camera info saved to:\n  {output_file}")

    # Release camera
    cam.release()

def main():
    save_camera_info(
        r"C:\Users\erlin\Skole høstsemester 2025\Ikt213g24h\assignments\solutions\Assignment_1\Løsning oppgave 1 web kamera"
    )

if __name__ == "__main__":
    main()

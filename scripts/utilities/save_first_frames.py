import cv2
import os

def export_first_frames(input_folder):
    # Create the output folder if it doesn't exist
    output_folder = os.path.join(input_folder, "frames")
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp4"):
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{video_name}.png")

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"⏭️  Skipping {filename} (already exported)")
                continue

            # Open the video and read the first frame
            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()
            if success:
                cv2.imwrite(output_path, frame)
                print(f"✅ Saved first frame of {filename} → {output_path}")
            else:
                print(f"⚠️ Could not read frame from {filename}")

            cap.release()

if __name__ == "__main__":
    folder = "footage/selection_finale/UPSCALED"
    export_first_frames(folder)

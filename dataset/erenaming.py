import os
import shutil

# Define source and destination directories
source_root = r"D:\Omkar Sonawale\data\models\HD\pro"
destination_folder = r"D:\Omkar Sonawale\data\models\HD\dataset\HR"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# Counter for renaming
counter = 1
print("renaming images........")
# Traverse all subdirectories
for subdir, _, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith(image_extensions):
            src_path = os.path.normpath(os.path.join(subdir, file))

            if not os.path.exists(src_path):
                print(f"Source file not found: {src_path}")
                continue

            # ext = os.path.splitext(file)[1]
            ext=".png"
            new_name = f"img_{counter:05d}{ext}"
            dst_path = os.path.join(destination_folder, new_name)

            try:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                counter += 1

                if counter == 15000:
                    break

            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")


print(f"Copied and renamed {counter - 1} images to '{destination_folder}'.")

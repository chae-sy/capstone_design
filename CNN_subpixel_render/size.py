import os
from PIL import Image

def print_image_sizes(directory):
    image_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])

    if not image_files:
        print("📂 No image files found in directory.")
        return

    print(f"📁 Directory: {directory}")
    print(f"🖼️ Total images: {len(image_files)}\n")

    for filename in image_files:
        path = os.path.join(directory, filename)
        try:
            with Image.open(path) as img:
                width, height = img.size
                print(f"{filename}: {width} × {height}")
        except Exception as e:
            print(f"⚠️ Could not open {filename}: {e}")

# 예시 사용
# 바꾸고 싶은 경로를 여기에 넣으세요
print_image_sizes('data/DIV2K_train_HR')


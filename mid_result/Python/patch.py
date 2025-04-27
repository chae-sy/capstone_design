import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Paths
source_dir = './data/DIV2K_train_HR'    # Your full DIV2K training images
save_dir = './data/DIV2K_patches_4800'   # Where to save patches
os.makedirs(save_dir, exist_ok=True)

# Settings
patch_size = 100
patches_per_image = 6

# Get all DIV2K images
image_files = sorted([
    f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg'))
])

# Transformation: random crop only
random_crop = transforms.RandomCrop(patch_size)

patch_counter = 0

for img_name in tqdm(image_files, desc="Processing DIV2K images"):
    img_path = os.path.join(source_dir, img_name)
    img = Image.open(img_path).convert('RGB')

    if img.width < patch_size or img.height < patch_size:
        print(f"Warning: {img_name} is too small for cropping, skipping.")
        continue

    for i in range(patches_per_image):
        patch = random_crop(img)
        save_path = os.path.join(save_dir, f"{patch_counter+1:05d}.png")
        patch.save(save_path)
        patch_counter += 1

print(f"✅ Done! Total patches created: {patch_counter}")


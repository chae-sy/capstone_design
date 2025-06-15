import os
from torchvision import transforms
from PIL import Image
import random

# 원본 데이터 경로
source_dir = 'data/DIV2K_train_HR'
# 저장할 augmented 데이터 경로
augmented_dir = 'data/DIV2K_train_HR_augmented_100x100'

# 디렉토리 없으면 생성
os.makedirs(augmented_dir, exist_ok=True)

# 원본 transform (load용)
base_transform = transforms.ToTensor()
image_size =100
# augmentation 정의
augmentations = [
    transforms.RandomCrop((image_size, image_size)),  # Crop to 256x256 patches
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=90)),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=180)),
    transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=270)),
]

# 원본 이미지 리스트
image_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(source_dir, img_name)
    img = Image.open(img_path)

    # 원본 이미지도 복사
    img.save(os.path.join(augmented_dir, img_name))

    # 각 augmentation 적용 후 저장
    for idx, aug in enumerate(augmentations):
        aug_img = aug(img)
        new_name = img_name.split('.')[0] + f'_aug{idx}.png'
        aug_img.save(os.path.join(augmented_dir, new_name))


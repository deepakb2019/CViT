# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import os
# import numpy as np
# import torch
# from PIL import Image

# import random

# # Define the transformations
# # data_transforms = transforms.Compose([
# #     transforms.Resize((224, 224)),  # Resize to the input size
# #     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Affine transformation
# #     transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
# #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
# #     transforms.ToTensor(),  # Convert to tensor
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
# # ])
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to the input size
#     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Affine transformation
#     transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
#     transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.8, 1.0))], p=0.5),  # Random resized crop
#     transforms.RandomApply([transforms.Lambda(
#         lambda img: random_cut_out(np.array(img), size=(50, 50))
#     )], p=0.5),  # Random cut-out
#     transforms.RandomApply([transforms.Lambda(
#         lambda img: face_cut_out(np.array(img))
#     )], p=0.5),  # Face cut-out
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
# ])

# def random_cut_out(image, size=(50, 50)):
#     """Randomly cuts out a square from the image."""
#     h, w, _ = image.shape
#     x = np.random.randint(0, w - size[0])
#     y = np.random.randint(0, h - size[1])
#     image[y:y + size[1], x:x + size[0], :] = 0
#     return Image.fromarray(image)

# def face_cut_out(image):
#     """Cuts out a random region mimicking a face cut-out."""
#     h, w, _ = image.shape
#     x1, y1 = w // 4, h // 4
#     x2, y2 = 3 * w // 4, 3 * h // 4
#     image[y1:y2, x1:x2, :] = 0
#     return Image.fromarray(image)


# # Load sample data
# data_dir = "data"
# dataset = ImageFolder(data_dir, transform=data_transforms)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# print("dataloader",dataloader)

# # Utility to denormalize images for visualization
# def denormalize(image):
#     """Denormalizes a tensor image to [0, 1] range for visualization."""
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     image = image * std + mean
#     return torch.clamp(image, 0, 1)

# # Visualize a batch of augmented images
# # Visualize a batch of augmented images
# def visualize_batch(dataloader):
#     """Visualizes a batch of images with labels."""
#     images, labels = next(iter(dataloader))
#     images = [denormalize(img).permute(1, 2, 0).numpy() for img in images]
#     plt.figure(figsize=(12, 8))
#     for idx, img in enumerate(images):
#         plt.subplot(1, len(images), idx + 1)
#         plt.imshow(img)
#         plt.axis("off")
#         plt.title(f"Label: {labels[idx].item()}")
#     plt.show()

# # Visualize augmented data
# visualize_batch(dataloader)


import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import random
import os

# Custom transformations
def random_cut_out(image, size=(50, 50)):
    """Randomly cuts out a square from the image."""
    h, w, _ = image.shape
    x = np.random.randint(0, w - size[0])
    y = np.random.randint(0, h - size[1])
    image[y:y + size[1], x:x + size[0], :] = 0
    return Image.fromarray(image)

def face_cut_out(image):
    """Cuts out a random region mimicking a face cut-out."""
    h, w, _ = image.shape
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    image[y1:y2, x1:x2, :] = 0
    return Image.fromarray(image)

# Define augmentations for visualization
transform_affine = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

transform_random_cutout = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure all images are resized
    transforms.Lambda(lambda img: random_cut_out(np.array(img), size=(50, 50))),
    transforms.ToTensor()
])

transform_face_cutout = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure all images are resized
    transforms.Lambda(lambda img: face_cut_out(np.array(img))),
    transforms.ToTensor()
])

# Load sample images
data_dir = "data"  # Replace with your dataset path
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory {data_dir} not found!")

dataset = ImageFolder(data_dir, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(dataloader))

# Prepare visualization grid
augmented_images = {
    "Affine": [transform_affine(Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8) * 255)) for img in images],
    "Random Cut-Out": [transform_random_cutout(Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8) * 255)) for img in images],
    "Face Cut-Out": [transform_face_cutout(Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8) * 255)) for img in images],
}

# Plot results
fig, axes = plt.subplots(3, len(images), figsize=(15, 5))
fig.subplots_adjust(hspace=0.5)

for row_idx, (aug_type, aug_images) in enumerate(augmented_images.items()):
    for col_idx, img in enumerate(aug_images):
        ax = axes[row_idx, col_idx]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(aug_type, fontsize=12)

plt.show()

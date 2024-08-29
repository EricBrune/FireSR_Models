import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split="val"):
    # horizontal flip OR rotate
    hflip = hflip and (split == "train" and random.random() < 0.5)
    vflip = rot and (split == "train" and random.random() < 0.5)
    rot90 = rot and (split == "train" and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()


def transform_augment(img_list, mask, split="val", min_max=(0, 1)):
    transformed_imgs = []
    for img in img_list:
        img_tensor = torch.nan_to_num(totensor(img))
        if split == "train":
            img_tensor = hflip(img_tensor.unsqueeze(0)).squeeze(0)
        img_tensor = img_tensor * (min_max[1] - min_max[0]) + min_max[0]
        transformed_imgs.append(img_tensor)
    
    # Transform the mask similarly
    mask = torch.nan_to_num(totensor(mask))
    if split == "train":
        mask = hflip(mask.unsqueeze(0)).squeeze(0)
    mask = mask * (min_max[1] - min_max[0]) + min_max[0]
    
    transformed_imgs.append(mask)
    return transformed_imgs


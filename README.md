
# FireSR Models

## Overview

This repository contains code for training and testing models for super-resolution and segmentation of wildfire-burned areas, as presented in the manuscript "FireSR: A Dataset for Super-Resolution and Segmentation of Burned Areas", submitted to NeurIPS 2024 Datasets and Benchmarks Track. This repository includes three main directories:

- **FiRes-DDPM**: A multitask adaptation of the Image-Super-Resolution-via-Iterative-Refinement (SR3) model, generating both super-resolved images and segmentation masks.
- **Image-Super-Resolution-via-Iterative-Refinement**: An implementation of the SR3 model for single-image super-resolution. [Original repo](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/)
- **Pytorch-UNet**: A U-Net implementation for semantic segmentation tasks. [Original repo](https://github.com/milesial/Pytorch-UNet)

## Getting Started

### Step 1: Prepare Your Environment

Before using the models, ensure you have the necessary dependencies installed. You can do this by running the following commands in each model directory:

```bash
pip install -r requirements.txt
```

### Step 2: Tiling the GeoTIFF Files

To train or test the models, you need to prepare your dataset by tiling the GeoTIFF files into smaller patches. This step is crucial for handling high-resolution imagery efficiently. Follow the steps below to tile your images:


1. **Run the Tiling Script**

   Use the Python script below to tile the GeoTIFF images. This script divides each image into smaller patches of a specified size (e.g., 128x128 pixels).

   ```python
   import rasterio
   from rasterio.windows import Window
   import os

   def tile_image(image_path, output_dir, tile_size=128):
       with rasterio.open(image_path) as src:
           for i in range(0, src.height, tile_size):
               for j in range(0, src.width, tile_size):
                   window = Window(j, i, tile_size, tile_size)
                   transform = src.window_transform(window)
                   outpath = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{i}_{j}.tif")
                   with rasterio.open(outpath, 'w', driver='GTiff', height=tile_size, width=tile_size, count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=transform) as dst:
                       dst.write(src.read(window=window))

   # Example usage
   image_dir = 'FireSR/dataset/S2/post/'
   output_dir = 'FireSR_tiled/hr_128/'
   os.makedirs(output_dir, exist_ok=True)

   for image_file in os.listdir(image_dir):
       if image_file.endswith('.tif'):
           tile_image(os.path.join(image_dir, image_file), output_dir)
   ```

   Make sure to adjust the `image_dir` and `output_dir` to point to your specific directories.

2. **Dataset structure**

   Your dataset should be structured in the following format:

   ```
   FireSR_train/
   │
   ├── Daymet_128/
   ├── LULC_128/
   ├── hr_128/
   ├── hr_mask_128/  # Only for FiRes-DDPM
   ├── lr_16/
   ├── pre_fire_128/
   ├── sr_16_128/
   ```

### Step 3: Training the Models

#### FiRes-DDPM

1. Navigate to the `FiRes-DDPM` directory.

2. Follow the instructions in the model's README to set up the configuration files and initiate training:

   ```bash
   python sr.py -p train -c config/train_S2_MODIS.json
   ```

#### Image-Super-Resolution-via-Iterative-Refinement

1. Navigate to the `Image-Super-Resolution-via-Iterative-Refinement` directory.

2. Edit the dataset configuration in the JSON files to match your data paths and resolutions.

3. Start training:

   ```bash
   python sr.py -p train -c config/train_S2_MODIS.json
   ```

#### Pytorch-UNet

1. Navigate to the `Pytorch-UNet` directory.

2. To train the model, run:

   ```bash
   python train.py --epochs 5 --batch-size 16 --learning-rate 0.001 --amp
   ```

### Step 4: Testing the models

#### FiRes-DDPM

1. Navigate to the `FiRes-DDPM` directory.

2. Follow the instructions in the model's README to set up the configuration files and initiate training:

   ```bash
   python sr.py -p val -c config/test_S2_MODIS.json
   ```

#### Image-Super-Resolution-via-Iterative-Refinement

1. Navigate to the `Image-Super-Resolution-via-Iterative-Refinement` directory.

2. Edit the dataset configuration in the JSON files to match your data paths and resolutions.

3. Start training:

   ```bash
   python sr.py -p val -c config/test_S2_MODIS.json
   ```

#### Pytorch-UNet

1. Navigate to the `Pytorch-UNet` directory.

2. To train the model, run:

   ```bash
   python predict.py -i1 /path/to/post/imgs -i2 /path/to/pre/imgs -o /path/to/output/dir
   ```

## Model Weights

Pre-trained model weights can be downloaded from the following link:

[Download Model Weights](https://drive.google.com/drive/folders/1qARZM6klJvf8IftqxfEX7a09zCV2Dk1d?usp=sharing)


## License

This repository is licensed under the MIT License.

## Contact

For any questions or further information, please contact:
- Name: Eric Brune
- Email: ebrune@kth.se

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import glob
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net, full_img1, full_img2, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img1 = torch.from_numpy(BasicDataset.preprocess(None, full_img1, scale_factor, is_mask=False))
    img2 = torch.from_numpy(BasicDataset.preprocess(None, full_img2, scale_factor, is_mask=False))
    
    # Assuming both images have same number of channels
    img = torch.cat((img1, img2), dim=0)
    
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img1.size[1], full_img1.size[0]), mode='bilinear')
        if net.n_classes > 1:
            print(output.shape)
            mask = torch.sigmoid(output[:, 1]) > out_threshold
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input_dir1', '-i1', metavar='INPUT_DIR1', help='Directory of post-fire input images', required=True)
    parser.add_argument('--input_dir2', '-i2', metavar='INPUT_DIR2', help='Directory of pre-fire input images', required=True)

    parser.add_argument('--output', '-o', default = "outputs/", metavar='OUTPUT', help='Directory of output images')
    parser.add_argument('--viz', '-v', dest='viz', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    input_dir1 = args.input_dir1
    input_dir2 = args.input_dir2

    files1 = sorted(glob.glob(f"{input_dir1}/*.png"))
    files2 = sorted(glob.glob(f"{input_dir2}/*.png"))
    output_dir = args.output  # We now consider it a directory
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    if os.path.isdir(files1[0]):
        in_files = glob.glob(files1[0] + "/*")
        out_files = ["outputs/" + file.split("/")[-1] for file in in_files]

    net = UNet(n_channels=6, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 255])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    for i, (fn1, fn2) in enumerate(zip(files1, files2)):
        img1 = Image.open(fn1)
        img2 = Image.open(fn2)

        mask = predict_img(net=net,
                           full_img1=img1,
                           full_img2=img2,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = os.path.join(output_dir, os.path.basename(fn1))  # Creates new filename based on input filename
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {fn1}, close to continue...')
            plot_img_and_mask(img1, mask)

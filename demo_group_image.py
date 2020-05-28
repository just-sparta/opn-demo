# -*- coding: utf-8 -*-

from __future__ import division
import torch
import torch.nn as nn

# general libs
import cv2
from PIL import Image
import os
import sys
import argparse

### My libs
sys.path.append('utils/')
sys.path.append('models/')
from utils.helpers import *
from models.OPN import OPN


def get_arguments():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--input_images_path", type=str, required=True)
    parser.add_argument("--image_w", type=int, required=True)
    parser.add_argument("--image_h", type=int, required=True)
    parser.add_argument("--image_count", type=int, required=True)
    parser.add_argument("--input_masks_path", type=str, required=True)
    return parser.parse_args()


args = get_arguments()
T = args.image_count
H = args.image_h
W = args.image_w
image_path = args.input_images_path
mask_path = args.input_masks_path
seq_name = os.path.basename(os.path.normpath(image_path))

#################### Load image
frames = np.empty((T, H, W, 3), dtype=np.float32)
holes = np.empty((T, H, W, 1), dtype=np.float32)
dists = np.empty((T, H, W, 1), dtype=np.float32)

images = sorted(os.listdir(image_path))
print(images)
masks = sorted(os.listdir(mask_path))
print(masks)

for index in range(T):
    #### rgb
    img_file = os.path.join(image_path, '{}'.format(images[index]))
    print(images[index])
    raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
    raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    frames[index] = raw_frame
    #### mask
    mask_file = os.path.join(mask_path, '{}'.format(masks[index]))
    print(masks[index])
    raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
    raw_mask = (raw_mask > 0.5).astype(np.uint8)
    raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
    raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    holes[index, :, :, 0] = raw_mask.astype(np.float32)
    #### dist
    dists[index, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)

frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
# remove hole
frames = frames * (1 - holes) + holes * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
# valids area
valids = 1 - holes
# unsqueeze to batch 1
frames = frames.unsqueeze(0)
holes = holes.unsqueeze(0)
dists = dists.unsqueeze(0)
valids = valids.unsqueeze(0)

#################### Load Model
model = nn.DataParallel(OPN())
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(os.path.join('OPN.pth')), strict=False)
print('Weights loading')
model.eval()

################### Inference
# memory encoding
midx = list(range(0, T))
with torch.no_grad():
    mkey, mval, mhol = model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])

for f in range(T):
    # memory selection
    ridx = [i for i in range(len(midx)) if i != f]  # memory minus self
    fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
    # inpainting..
    for r in range(999):
        if r == 0:
            comp = frames[:, :, f]
            dist = dists[:, :, f]
        with torch.no_grad():
            comp, dist = model(fkey, fval, fhol, comp, valids[:, :, f], dist)

        # update
        comp, dist = comp.detach(), dist.detach()
        if torch.sum(dist).item() == 0:
            break

    # visualize..
    est = (comp[0].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
    true = (frames[0, :, f].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)  # h,w,3
    mask = (dists[0, 0, f].detach().cpu().numpy() > 0).astype(np.uint8)  # h,w,1
    ov_true = overlay_davis(true, mask, colors=[[0, 0, 0], [100, 100, 0]], cscale=2, alpha=0.4)

    save_path = os.path.join('Image_results', seq_name, 'concat')
    save_path_for_img = os.path.join('Image_results', seq_name, 'result')

    canvas = np.concatenate([ov_true, est], axis=0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    canvas = Image.fromarray(canvas)
    canvas.save(os.path.join(save_path, 'res_{}.jpg'.format(f)))

    if not os.path.exists(save_path_for_img):
        os.makedirs(save_path_for_img)

    est = Image.fromarray(est)
    est.save(os.path.join(save_path_for_img, 'est_{}.jpg'.format(f)))

print('Results are saved: ./{}'.format(save_path))

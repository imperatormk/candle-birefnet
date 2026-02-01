#!/usr/bin/env python3
"""Debug squeeze_module step by step."""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '/Users/zimski/projects/oss/BiRefNet')

from models.birefnet import BiRefNet

def main():
    device = 'mps'

    # Load model
    print("Loading BiRefNet...")
    model = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
    model = model.to(device).eval()

    # Load and preprocess image
    img_path = '/Users/zimski/Downloads/zimage_flash.png'
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1024, 1024), Image.LANCZOS)

    # To tensor with ImageNet normalization
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std

    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Get features as model does
        x1, x2, x3, x4 = model.bb(x)

        # mul_scl_ipt
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x1_, x2_, x3_, x4_ = model.bb(x_half)

        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)

        # cxt
        x1_to_x4 = F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x2_to_x4 = F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x3_to_x4 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x4_cxt = torch.cat([x1_to_x4, x2_to_x4, x3_to_x4, x4], dim=1)

        print(f"x4_cxt: {x4_cxt.shape}, range: [{x4_cxt.min():.4f}, {x4_cxt.max():.4f}]")

        # squeeze_module step by step
        squeeze = model.squeeze_module[0]

        # conv_in
        out = squeeze.conv_in(x4_cxt)
        print(f"after conv_in: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")

        # bn_in
        out = squeeze.bn_in(out)
        print(f"after bn_in: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")

        # relu_in
        out = squeeze.relu_in(out)
        print(f"after relu_in: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")

        # dec_att (ASPP)
        if hasattr(squeeze, 'dec_att'):
            out = squeeze.dec_att(out)
            print(f"after dec_att: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")

        # conv_out
        out = squeeze.conv_out(out)
        print(f"after conv_out: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")

        # bn_out
        out = squeeze.bn_out(out)
        print(f"after bn_out: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")

        # Full squeeze output
        full_out = model.squeeze_module(x4_cxt)
        print(f"\nFull squeeze_module output: {full_out.shape}, range: [{full_out.min():.4f}, {full_out.max():.4f}]")

if __name__ == '__main__':
    main()

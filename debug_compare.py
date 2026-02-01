#!/usr/bin/env python3
"""Compare Python BiRefNet intermediate outputs to debug Rust implementation."""

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
    print(f"Input shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")

    with torch.no_grad():
        # Get backbone features
        print("\n=== Backbone ===")
        x1, x2, x3, x4 = model.bb(x)
        print(f"x1: {x1.shape}, range: [{x1.min():.4f}, {x1.max():.4f}]")
        print(f"x2: {x2.shape}, range: [{x2.min():.4f}, {x2.max():.4f}]")
        print(f"x3: {x3.shape}, range: [{x3.min():.4f}, {x3.max():.4f}]")
        print(f"x4: {x4.shape}, range: [{x4.min():.4f}, {x4.max():.4f}]")

        # mul_scl_ipt
        print("\n=== mul_scl_ipt (half scale) ===")
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x1_, x2_, x3_, x4_ = model.bb(x_half)

        x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        print(f"x1 after cat: {x1.shape}")
        print(f"x2 after cat: {x2.shape}")
        print(f"x3 after cat: {x3.shape}")
        print(f"x4 after cat: {x4.shape}")

        # cxt concatenation
        print("\n=== cxt (context features) ===")
        x1_to_x4 = F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x2_to_x4 = F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x3_to_x4 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x4_cxt = torch.cat([x1_to_x4, x2_to_x4, x3_to_x4, x4], dim=1)
        print(f"x4 after cxt: {x4_cxt.shape}, range: [{x4_cxt.min():.4f}, {x4_cxt.max():.4f}]")

        # squeeze_module
        print("\n=== squeeze_module ===")
        x4_squeezed = model.squeeze_module(x4_cxt)
        print(f"x4 after squeeze: {x4_squeezed.shape}, range: [{x4_squeezed.min():.4f}, {x4_squeezed.max():.4f}]")

        # Full forward to get output
        print("\n=== Full forward ===")
        output = model(x)
        if isinstance(output, list):
            output = output[-1]
        print(f"Output shape: {output.shape}")
        print(f"Output range (before sigmoid): [{output.min():.4f}, {output.max():.4f}]")
        output_sig = torch.sigmoid(output)
        print(f"Output range (after sigmoid): [{output_sig.min():.4f}, {output_sig.max():.4f}]")

        # Save output
        mask = output_sig.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save('/Users/zimski/Downloads/zimage_flash_mask_python.png')
        print("\nSaved Python mask to ~/Downloads/zimage_flash_mask_python.png")

if __name__ == '__main__':
    main()

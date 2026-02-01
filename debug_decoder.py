#!/usr/bin/env python3
"""Debug decoder step by step."""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '/Users/zimski/projects/oss/BiRefNet')

from models.birefnet import BiRefNet, image2patches

def main():
    device = 'mps'

    print("Loading BiRefNet...")
    model = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
    model = model.to(device).eval()

    img = Image.open('/Users/zimski/Downloads/zimage_flash.png').convert('RGB').resize((1024, 1024), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Get features through encoder
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

        # squeeze
        x4_sq = model.squeeze_module(x4_cxt)
        print(f"x4 after squeeze: {x4_sq.shape}, sum: {x4_sq.sum():.4f}")

        # Now decoder
        decoder = model.decoder

        # ipt_blk5
        patches5 = image2patches(x, patch_ref=x4_sq, transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
        print(f"patches5: {patches5.shape}, sum: {patches5.sum():.4f}")

        patches5_interp = F.interpolate(patches5, size=x4_sq.shape[2:], mode='bilinear', align_corners=True)
        print(f"patches5_interp: {patches5_interp.shape}, sum: {patches5_interp.sum():.4f}")

        ipt5 = decoder.ipt_blk5(patches5_interp)
        print(f"ipt5: {ipt5.shape}, sum: {ipt5.sum():.4f}")

        # concat x4 + ipt5
        d4_in = torch.cat([x4_sq, ipt5], dim=1)
        print(f"d4_in: {d4_in.shape}, sum: {d4_in.sum():.4f}")

        # decoder_block4
        p4 = decoder.decoder_block4(d4_in)
        print(f"p4: {p4.shape}, sum: {p4.sum():.4f}")

        # Continue through decoder...
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + decoder.lateral_block4(x3)
        print(f"_p3 (after lateral): {_p3.shape}, sum: {_p3.sum():.4f}")

        # ipt_blk4
        patches4 = image2patches(x, patch_ref=_p3, transformation='b c (hg h) (wg w) -> b (c hg wg) h w')
        patches4_interp = F.interpolate(patches4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        ipt4 = decoder.ipt_blk4(patches4_interp)
        print(f"ipt4: {ipt4.shape}, sum: {ipt4.sum():.4f}")

        _p3 = torch.cat([_p3, ipt4], dim=1)
        p3 = decoder.decoder_block3(_p3)
        print(f"p3: {p3.shape}, sum: {p3.sum():.4f}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Debug script to save Python Swin intermediate values at each stage.
This helps identify where Rust diverges from Python.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/Users/zimski/projects/oss/BiRefNet')

from models.backbones.swin_v1 import SwinTransformer, PatchEmbed, BasicLayer, WindowAttention

def save_npy(tensor, path):
    """Save tensor to numpy file."""
    arr = tensor.detach().cpu().numpy()
    np.save(path, arr)
    print(f"  Saved: {path}, shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}], sum={arr.sum():.4f}")

def debug_swin_stages():
    device = 'mps'

    # Load pretrained model
    print("Loading BiRefNet with Swin-L backbone...")
    from models.birefnet import BiRefNet
    model = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
    model = model.to(device).eval()

    # Get the Swin backbone
    swin = model.bb  # This is the SwinTransformer

    print(f"\nSwin config:")
    print(f"  embed_dim: {swin.embed_dim}")
    print(f"  num_layers: {swin.num_layers}")
    print(f"  num_features: {swin.num_features}")

    # Load input
    if True:  # Use the saved input
        x = np.load('/tmp/py_input.npy')
        x = torch.from_numpy(x).to(device)
        print(f"\nLoaded input: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")

    with torch.no_grad():
        # Save input
        save_npy(x, '/tmp/swin_input.npy')

        # === Patch Embedding ===
        print("\n=== Patch Embedding ===")
        x_pe = swin.patch_embed(x)
        save_npy(x_pe, '/tmp/swin_patch_embed_out.npy')

        Wh, Ww = x_pe.size(2), x_pe.size(3)
        print(f"  Wh={Wh}, Ww={Ww}")

        # Flatten and transpose for transformer layers
        x = x_pe.flatten(2).transpose(1, 2)
        x = swin.pos_drop(x)
        save_npy(x, '/tmp/swin_after_flatten.npy')

        print(f"  After flatten: shape={x.shape}")

        # === Process each layer ===
        for i in range(swin.num_layers):
            print(f"\n=== Layer {i} ===")
            layer = swin.layers[i]

            # Save input to this layer
            save_npy(x, f'/tmp/swin_layer{i}_input.npy')

            # Process layer
            x_out, H, W, x_down, Wh_new, Ww_new = layer(x, Wh, Ww)

            # Save x_out (before norm)
            save_npy(x_out, f'/tmp/swin_layer{i}_x_out.npy')

            # Apply norm and reshape
            norm_layer = getattr(swin, f'norm{i}')
            x_normed = norm_layer(x_out)
            save_npy(x_normed, f'/tmp/swin_layer{i}_normed.npy')

            out = x_normed.view(-1, H, W, swin.num_features[i]).permute(0, 3, 1, 2).contiguous()
            save_npy(out, f'/tmp/swin_layer{i}_out.npy')

            print(f"  H={H}, W={W}, out shape={out.shape}")
            print(f"  Wh_new={Wh_new}, Ww_new={Ww_new}")

            # Update for next layer
            x = x_down
            Wh, Ww = Wh_new, Ww_new

            # Debug first block of first layer
            if i == 0:
                debug_first_block(layer, swin, device)

def debug_first_block(layer, swin, device):
    """Debug the first block of the first layer in detail."""
    print("\n  === Debugging Block 0 in detail ===")

    # Reload input for detailed debugging
    x_input = np.load('/tmp/swin_layer0_input.npy')
    x = torch.from_numpy(x_input).to(device)

    H, W = 256, 256  # Input size / patch_size = 1024/4 = 256

    block = layer.blocks[0]
    block.H = H
    block.W = W

    B, L, C = x.shape
    print(f"  Input: B={B}, L={L}, C={C}, H={H}, W={W}")

    # Norm1
    shortcut = x
    x = block.norm1(x)
    save_npy(x, '/tmp/swin_block0_after_norm1.npy')

    x = x.view(B, H, W, C)

    # Padding
    pad_r = (block.window_size - W % block.window_size) % block.window_size
    pad_b = (block.window_size - H % block.window_size) % block.window_size
    print(f"  Padding: pad_r={pad_r}, pad_b={pad_b}")
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    _, Hp, Wp, _ = x.shape
    print(f"  After padding: Hp={Hp}, Wp={Wp}")
    save_npy(x, '/tmp/swin_block0_after_pad.npy')

    # Attention mask calculation (same as in BasicLayer.forward)
    window_size = layer.window_size
    shift_size = layer.shift_size

    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    from models.backbones.swin_v1 import window_partition
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float(0.0)).to(x.dtype)

    print(f"  Attention mask shape: {attn_mask.shape}")
    save_npy(attn_mask, '/tmp/swin_block0_attn_mask.npy')

    # Block 0 has shift_size=0 so no cyclic shift
    shifted_x = x

    # Window partition
    x_windows = window_partition(shifted_x, block.window_size)
    x_windows = x_windows.view(-1, block.window_size * block.window_size, C)
    save_npy(x_windows, '/tmp/swin_block0_windows.npy')
    print(f"  Windows shape: {x_windows.shape}")

    # Window attention
    print("\n  === Window Attention ===")
    attn = block.attn
    B_, N, C = x_windows.shape

    # QKV projection
    qkv = attn.qkv(x_windows)
    save_npy(qkv, '/tmp/swin_block0_qkv.npy')
    print(f"  QKV shape: {qkv.shape}")

    qkv = qkv.reshape(B_, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    save_npy(q, '/tmp/swin_block0_q.npy')
    save_npy(k, '/tmp/swin_block0_k.npy')
    save_npy(v, '/tmp/swin_block0_v.npy')
    print(f"  Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

    # Relative position bias
    relative_position_bias = attn.relative_position_bias_table[attn.relative_position_index.view(-1)].view(
        N, N, -1
    ).permute(2, 0, 1).unsqueeze(0).to(dtype=q.dtype, device=q.device)
    save_npy(relative_position_bias, '/tmp/swin_block0_rel_pos_bias.npy')
    print(f"  Relative position bias shape: {relative_position_bias.shape}")

    # Attention scores
    q_scaled = q * attn.scale
    attn_scores = q_scaled @ k.transpose(-2, -1)
    save_npy(attn_scores, '/tmp/swin_block0_attn_scores.npy')
    print(f"  Attention scores shape: {attn_scores.shape}, range=[{attn_scores.min():.4f}, {attn_scores.max():.4f}]")

    # Add relative position bias
    attn_with_bias = attn_scores + relative_position_bias
    save_npy(attn_with_bias, '/tmp/swin_block0_attn_with_bias.npy')

    # For block 0, no mask is applied (shift_size=0)
    # Softmax
    attn_probs = attn_with_bias.softmax(dim=-1)
    save_npy(attn_probs, '/tmp/swin_block0_attn_probs.npy')
    print(f"  Attention probs range: [{attn_probs.min():.4f}, {attn_probs.max():.4f}]")

    # Attention output
    attn_out = attn_probs @ v
    save_npy(attn_out, '/tmp/swin_block0_attn_out.npy')
    print(f"  Attention output shape: {attn_out.shape}")

    # Reshape and project
    attn_out = attn_out.transpose(1, 2).reshape(B_, N, C)
    attn_out = attn.proj(attn_out)
    save_npy(attn_out, '/tmp/swin_block0_proj_out.npy')
    print(f"  After proj: {attn_out.shape}")

if __name__ == '__main__':
    debug_swin_stages()

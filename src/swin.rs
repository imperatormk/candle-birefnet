//! Swin Transformer v1 for candle
//!
//! A hierarchical vision transformer using shifted windows.
//! Reference: https://arxiv.org/abs/2103.14030

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

#[cfg(feature = "flash-attn")]
use candle_mps_flash_attention::{flash_attention_with_bias, flash_attention_with_repeating_bias};

/// Configuration for Swin Transformer
#[derive(Clone, Debug)]
pub struct SwinConfig {
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub window_size: usize,
    pub mlp_ratio: f64,
    pub patch_size: usize,
    pub in_channels: usize,
    pub drop_path_rate: f64,
}

impl SwinConfig {
    /// Swin-T configuration
    pub fn swin_t() -> Self {
        Self {
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            patch_size: 4,
            in_channels: 3,
            drop_path_rate: 0.2,
        }
    }

    /// Swin-S configuration
    pub fn swin_s() -> Self {
        Self {
            embed_dim: 96,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            patch_size: 4,
            in_channels: 3,
            drop_path_rate: 0.2,
        }
    }

    /// Swin-B configuration
    pub fn swin_b() -> Self {
        Self {
            embed_dim: 128,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 12,
            mlp_ratio: 4.0,
            patch_size: 4,
            in_channels: 3,
            drop_path_rate: 0.2,
        }
    }

    /// Swin-L configuration (used by BiRefNet)
    pub fn swin_l() -> Self {
        Self {
            embed_dim: 192,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![6, 12, 24, 48],
            window_size: 12,
            mlp_ratio: 4.0,
            patch_size: 4,
            in_channels: 3,
            drop_path_rate: 0.2,
        }
    }

    /// Get output channels for each stage
    pub fn stage_channels(&self) -> Vec<usize> {
        (0..self.depths.len())
            .map(|i| self.embed_dim * (1 << i))
            .collect()
    }
}

/// MLP block
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_features, in_features, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

/// Window-based multi-head self attention
struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    relative_position_bias_table: Tensor,
    relative_position_index: Tensor,
    /// Pre-computed relative position bias [num_heads, N, N] for reuse
    cached_bias: Tensor,
    num_heads: usize,
    scale: f64,
    window_size: usize,
}

impl WindowAttention {
    fn new(
        dim: usize,
        window_size: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let qkv = candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = candle_nn::linear(dim, dim, vb.pp("proj"))?;

        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        // Relative position bias table: (2*Wh-1) * (2*Ww-1), num_heads
        let num_relative_positions = (2 * window_size - 1) * (2 * window_size - 1);
        let relative_position_bias_table = vb.get(
            (num_relative_positions, num_heads),
            "relative_position_bias_table",
        )?;

        // Build relative position index (matches PyTorch Swin implementation)
        let device = relative_position_bias_table.device();
        let relative_position_index = Self::build_relative_position_index(window_size, device)?;

        // Pre-compute the relative position bias [num_heads, N, N]
        let n_tokens = window_size * window_size;
        let index_flat = relative_position_index.flatten_all()?;
        let bias = relative_position_bias_table.index_select(&index_flat, 0)?;
        let bias = bias.reshape((n_tokens, n_tokens, num_heads))?;
        let cached_bias = bias.permute((2, 0, 1))?.contiguous()?; // [num_heads, N, N]

        Ok(Self {
            qkv,
            proj,
            relative_position_bias_table,
            relative_position_index,
            cached_bias,
            num_heads,
            scale,
            window_size,
        })
    }

    fn build_relative_position_index(window_size: usize, device: &Device) -> Result<Tensor> {
        let ws = window_size as i64;
        let _coords_h: Vec<i64> = (0..ws).collect();
        let _coords_w: Vec<i64> = (0..ws).collect();

        // Build coordinate grid
        let mut relative_coords = vec![0i64; (window_size * window_size * window_size * window_size) as usize];

        for i in 0..window_size {
            for j in 0..window_size {
                for k in 0..window_size {
                    for l in 0..window_size {
                        let idx = i * window_size * window_size * window_size
                            + j * window_size * window_size
                            + k * window_size
                            + l;
                        let rel_h = (i as i64) - (k as i64) + (ws - 1);
                        let rel_w = (j as i64) - (l as i64) + (ws - 1);
                        relative_coords[idx] = rel_h * (2 * ws - 1) + rel_w;
                    }
                }
            }
        }

        // Reshape to [Wh*Ww, Wh*Ww]
        let n = window_size * window_size;
        let mut index_flat = vec![0i64; n * n];
        for i in 0..window_size {
            for j in 0..window_size {
                for k in 0..window_size {
                    for l in 0..window_size {
                        let src_idx = i * window_size * window_size * window_size
                            + j * window_size * window_size
                            + k * window_size
                            + l;
                        let dst_row = i * window_size + j;
                        let dst_col = k * window_size + l;
                        index_flat[dst_row * n + dst_col] = relative_coords[src_idx];
                    }
                }
            }
        }

        Tensor::from_vec(index_flat, (n, n), device)
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_, n, c) = x.dims3()?;
        let head_dim = c / self.num_heads;

        // QKV projection: [B_, N, 3*C] -> [3, B_, num_heads, N, head_dim]
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b_, n, 3, self.num_heads, head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        // Try flash attention if available and on Metal
        #[cfg(feature = "flash-attn")]
        {
            let use_flash = std::env::var("DISABLE_FLASH_ATTN").is_err();
            if use_flash {
            if let candle_core::Device::Metal(_) = x.device() {
                // MFA adds bias to UNSCALED scores (S = Q*K^T), then scales during softmax
                // So we need to scale the bias by 1/scale = sqrt(head_dim) to match standard attention
                let inv_scale = 1.0 / self.scale;

                if let Some(window_mask) = mask {
                    // Shifted window attention: combine bias + window_mask, both scaled
                    let n_windows = window_mask.dim(0)?;
                    let bias = self.cached_bias.unsqueeze(0)?;
                    let window_mask = window_mask.unsqueeze(1)?;
                    let combined = bias.broadcast_add(&window_mask)?;
                    let scaled_combined = (&combined * inv_scale)?.contiguous()?;

                    let x = flash_attention_with_repeating_bias(&q, &k, &v, &scaled_combined, n_windows, false)?;
                    let x = x.transpose(1, 2)?;
                    let x = x.reshape((b_, n, c))?;
                    return self.proj.forward(&x);
                } else {
                    // Non-shifted window: just position bias, scaled
                    let bias = self.cached_bias.unsqueeze(0)?;
                    let scaled_bias = (&bias * inv_scale)?;

                    let x = flash_attention_with_bias(&q, &k, &v, &scaled_bias, false)?;
                    let x = x.transpose(1, 2)?;
                    let x = x.reshape((b_, n, c))?;
                    return self.proj.forward(&x);
                }
            }
            }
        }

        // Fallback: standard attention for non-Metal devices
        self.forward_standard(&q, &k, &v, &self.cached_bias, mask, b_, n, c)
    }


    fn forward_standard(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        bias: &Tensor,
        mask: Option<&Tensor>,
        b_: usize,
        n: usize,
        c: usize,
    ) -> Result<Tensor> {
        // Scale query
        let q = (q * self.scale)?;

        // Attention: Q @ K^T
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        // Add relative position bias
        let bias = bias.unsqueeze(0)?; // [1, num_heads, N, N]
        let attn = attn.broadcast_add(&bias)?;

        // Apply mask if provided
        let attn = if let Some(mask) = mask {
            // mask shape: [nW, N, N] -> need to reshape for broadcasting
            let n_windows = mask.dim(0)?;
            let attn = attn.reshape((b_ / n_windows, n_windows, self.num_heads, n, n))?;
            let mask = mask.unsqueeze(0)?.unsqueeze(2)?; // [1, nW, 1, N, N]
            let attn = attn.broadcast_add(&mask)?;
            attn.reshape((b_, self.num_heads, n, n))?
        } else {
            attn
        };

        // Softmax
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // Attention @ V
        let x = attn.matmul(v)?;

        // Reshape: [B_, num_heads, N, head_dim] -> [B_, N, C]
        let x = x.transpose(1, 2)?;
        let x = x.reshape((b_, n, c))?;

        // Output projection
        self.proj.forward(&x)
    }
}

/// Swin Transformer Block
struct SwinTransformerBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    window_size: usize,
    shift_size: usize,
}

impl SwinTransformerBlock {
    fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn = WindowAttention::new(dim, window_size, num_heads, vb.pp("attn"))?;
        let norm2 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?;

        let mlp_hidden_dim = (dim as f64 * mlp_ratio) as usize;
        let mlp = Mlp::new(dim, mlp_hidden_dim, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
            shift_size,
        })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        assert_eq!(l, h * w, "Input feature has wrong size");

        let shortcut = x.clone();
        let x = self.norm1.forward(x)?;
        let x = x.reshape((b, h, w, c))?;

        // Pad to multiples of window size
        let pad_r = (self.window_size - w % self.window_size) % self.window_size;
        let pad_b = (self.window_size - h % self.window_size) % self.window_size;

        let x = if pad_r > 0 || pad_b > 0 {
            x.pad_with_zeros(2, 0, pad_r)?.pad_with_zeros(1, 0, pad_b)?
        } else {
            x
        };

        let (_, hp, wp, _) = x.dims4()?;

        // Cyclic shift
        let (shifted_x, use_mask) = if self.shift_size > 0 {
            // Roll along H and W dimensions
            let shifted = Self::roll_2d(&x, -(self.shift_size as i64), -(self.shift_size as i64))?;
            (shifted, true)
        } else {
            (x, false)
        };

        // Partition windows: [B, Hp, Wp, C] -> [B*nW, window_size*window_size, C]
        let x_windows = self.window_partition(&shifted_x)?;

        // Window attention
        let mask = if use_mask { attn_mask } else { None };
        let attn_windows = self.attn.forward(&x_windows, mask)?;

        // Merge windows
        let shifted_x = self.window_reverse(&attn_windows, hp, wp)?;

        // Reverse cyclic shift
        let x = if self.shift_size > 0 {
            Self::roll_2d(&shifted_x, self.shift_size as i64, self.shift_size as i64)?
        } else {
            shifted_x
        };

        // Remove padding
        let x = if pad_r > 0 || pad_b > 0 {
            x.narrow(1, 0, h)?.narrow(2, 0, w)?
        } else {
            x
        };

        let x = x.reshape((b, h * w, c))?;

        // Residual connection + FFN
        let x = (shortcut + x)?;
        let x = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;

        Ok(x)
    }

    fn roll_2d(x: &Tensor, shift_h: i64, shift_w: i64) -> Result<Tensor> {
        let (_b, h, w, _c) = x.dims4()?;
        let h = h as i64;
        let w = w as i64;

        // Normalize shifts to positive
        let shift_h = ((shift_h % h) + h) % h;
        let shift_w = ((shift_w % w) + w) % w;

        if shift_h == 0 && shift_w == 0 {
            return Ok(x.clone());
        }

        // Roll along H dimension
        let x = if shift_h > 0 {
            let part1 = x.narrow(1, (h - shift_h) as usize, shift_h as usize)?;
            let part2 = x.narrow(1, 0, (h - shift_h) as usize)?;
            Tensor::cat(&[&part1, &part2], 1)?
        } else {
            x.clone()
        };

        // Roll along W dimension
        let x = if shift_w > 0 {
            let part1 = x.narrow(2, (w - shift_w) as usize, shift_w as usize)?;
            let part2 = x.narrow(2, 0, (w - shift_w) as usize)?;
            Tensor::cat(&[&part1, &part2], 2)?
        } else {
            x
        };

        Ok(x)
    }

    fn window_partition(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let ws = self.window_size;

        // Reshape: [B, H, W, C] -> [B, H/ws, ws, W/ws, ws, C]
        let x = x.reshape((b, h / ws, ws, w / ws, ws, c))?;

        // Permute: -> [B, H/ws, W/ws, ws, ws, C]
        let x = x.permute((0, 1, 3, 2, 4, 5))?;

        // Reshape: -> [B * nW, ws*ws, C]
        let num_windows = (h / ws) * (w / ws);
        x.reshape((b * num_windows, ws * ws, c))
    }

    fn window_reverse(&self, windows: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let ws = self.window_size;
        let (b_nw, _, c) = windows.dims3()?;
        let num_windows = (h / ws) * (w / ws);
        let b = b_nw / num_windows;

        // Reshape: [B*nW, ws*ws, C] -> [B, H/ws, W/ws, ws, ws, C]
        let x = windows.reshape((b, h / ws, w / ws, ws, ws, c))?;

        // Permute: -> [B, H/ws, ws, W/ws, ws, C]
        let x = x.permute((0, 1, 3, 2, 4, 5))?;

        // Reshape: -> [B, H, W, C]
        x.reshape((b, h, w, c))
    }
}

/// Patch Merging layer (downsampling)
struct PatchMerging {
    reduction: Linear,
    norm: LayerNorm,
}

impl PatchMerging {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::layer_norm(4 * dim, 1e-5, vb.pp("norm"))?;
        let reduction = candle_nn::linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?;
        Ok(Self { reduction, norm })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, _, c) = x.dims3()?;
        let x = x.reshape((b, h, w, c))?;

        // Pad if needed
        let (x, h, w) = if h % 2 == 1 || w % 2 == 1 {
            let pad_w = w % 2;
            let pad_h = h % 2;
            let x_padded = x.pad_with_zeros(2, 0, pad_w)?.pad_with_zeros(1, 0, pad_h)?;
            (x_padded, h + pad_h, w + pad_w)
        } else {
            (x, h, w)
        };

        // Strided sampling: x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], etc.
        // Reshape to [B, H/2, 2, W/2, 2, C] then select indices
        let x = x.reshape((b, h / 2, 2, w / 2, 2, c))?;

        // x0 = x[:, 0::2, 0::2, :] = x[:, :, 0, :, 0, :]
        let x0 = x.narrow(2, 0, 1)?.narrow(4, 0, 1)?.squeeze(4)?.squeeze(2)?;
        // x1 = x[:, 1::2, 0::2, :] = x[:, :, 1, :, 0, :]
        let x1 = x.narrow(2, 1, 1)?.narrow(4, 0, 1)?.squeeze(4)?.squeeze(2)?;
        // x2 = x[:, 0::2, 1::2, :] = x[:, :, 0, :, 1, :]
        let x2 = x.narrow(2, 0, 1)?.narrow(4, 1, 1)?.squeeze(4)?.squeeze(2)?;
        // x3 = x[:, 1::2, 1::2, :] = x[:, :, 1, :, 1, :]
        let x3 = x.narrow(2, 1, 1)?.narrow(4, 1, 1)?.squeeze(4)?.squeeze(2)?;

        // Concatenate: [B, H/2, W/2, 4*C]
        let x = Tensor::cat(&[&x0, &x1, &x2, &x3], D::Minus1)?;

        // Reshape: [B, H/2 * W/2, 4*C]
        let x = x.reshape((b, (h / 2) * (w / 2), 4 * c))?;

        // Norm + Linear
        let x = self.norm.forward(&x)?;
        self.reduction.forward(&x)
    }
}

/// Basic layer (one stage)
pub struct BasicLayer {
    blocks: Vec<SwinTransformerBlock>,
    downsample: Option<PatchMerging>,
    window_size: usize,
    shift_size: usize,
}

impl BasicLayer {
    fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f64,
        downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let shift_size = window_size / 2;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block_shift = if i % 2 == 0 { 0 } else { shift_size };
            let block = SwinTransformerBlock::new(
                dim,
                num_heads,
                window_size,
                block_shift,
                mlp_ratio,
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        let downsample = if downsample {
            Some(PatchMerging::new(dim, vb.pp("downsample"))?)
        } else {
            None
        };

        Ok(Self {
            blocks,
            downsample,
            window_size,
            shift_size,
        })
    }

    pub fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<(Tensor, usize, usize, Tensor, usize, usize)> {
        // Calculate padded dimensions
        let hp = ((h + self.window_size - 1) / self.window_size) * self.window_size;
        let wp = ((w + self.window_size - 1) / self.window_size) * self.window_size;

        // Build attention mask for shifted window attention
        let attn_mask = self.create_attention_mask(hp, wp, x.device(), x.dtype())?;

        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, h, w, Some(&attn_mask))?;
        }

        let x_out = x.clone();

        let (x_down, wh, ww) = if let Some(ref downsample) = self.downsample {
            let down = downsample.forward(&x, h, w)?;
            (down, (h + 1) / 2, (w + 1) / 2)
        } else {
            (x, h, w)
        };

        Ok((x_out, h, w, x_down, wh, ww))
    }

    fn create_attention_mask(&self, hp: usize, wp: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        // Create mask for SW-MSA
        let mut img_mask = vec![0.0f32; hp * wp];

        // Define slices for the 9 regions
        let h_slices = [
            (0, hp - self.window_size),
            (hp - self.window_size, hp - self.shift_size),
            (hp - self.shift_size, hp),
        ];
        let w_slices = [
            (0, wp - self.window_size),
            (wp - self.window_size, wp - self.shift_size),
            (wp - self.shift_size, wp),
        ];

        let mut cnt = 0;
        for (h_start, h_end) in &h_slices {
            for (w_start, w_end) in &w_slices {
                for i in *h_start..*h_end {
                    for j in *w_start..*w_end {
                        img_mask[i * wp + j] = cnt as f32;
                    }
                }
                cnt += 1;
            }
        }

        let img_mask = Tensor::from_vec(img_mask, (1, hp, wp, 1), device)?;

        // Partition into windows
        let ws = self.window_size;
        let nw_h = hp / ws;
        let nw_w = wp / ws;
        let num_windows = nw_h * nw_w;

        // [1, nw_h, ws, nw_w, ws, 1] -> [nw_h*nw_w, ws*ws]
        let mask = img_mask.reshape((1, nw_h, ws, nw_w, ws, 1))?;
        let mask = mask.permute((0, 1, 3, 2, 4, 5))?;
        let mask = mask.reshape((num_windows, ws * ws))?;

        // Compute attention mask: [nW, N, N]
        let mask_1 = mask.unsqueeze(1)?;
        let mask_2 = mask.unsqueeze(2)?;
        let attn_mask = mask_1.broadcast_sub(&mask_2)?;

        // Where attn_mask != 0, set to -100, else 0
        let zeros = Tensor::zeros_like(&attn_mask)?;
        let neg_inf = (Tensor::ones_like(&attn_mask)? * (-100.0))?;
        let attn_mask = attn_mask.ne(0.0)?.where_cond(&neg_inf, &zeros)?;

        attn_mask.to_dtype(dtype)
    }
}

/// Patch Embedding layer
pub struct PatchEmbed {
    pub proj: Conv2d,
    pub norm: Option<LayerNorm>,
    patch_size: usize,
}

impl PatchEmbed {
    pub fn new(
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_channels, embed_dim, patch_size, conv_cfg, vb.pp("proj"))?;

        let norm = if norm {
            Some(candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("norm"))?)
        } else {
            None
        };

        Ok(Self {
            proj,
            norm,
            patch_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;

        // Pad if needed
        let x = if w % self.patch_size != 0 || h % self.patch_size != 0 {
            let pad_w = (self.patch_size - w % self.patch_size) % self.patch_size;
            let pad_h = (self.patch_size - h % self.patch_size) % self.patch_size;
            x.pad_with_zeros(3, 0, pad_w)?.pad_with_zeros(2, 0, pad_h)?
        } else {
            x.clone()
        };

        let x = self.proj.forward(&x)?;

        if let Some(ref norm) = self.norm {
            let (b, c, wh, ww) = x.dims4()?;
            let x = x.flatten_from(2)?.transpose(1, 2)?;
            let x = norm.forward(&x)?;
            x.transpose(1, 2)?.reshape((b, c, wh, ww))
        } else {
            Ok(x)
        }
    }
}

/// Swin Transformer backbone
pub struct SwinTransformer {
    pub patch_embed: PatchEmbed,
    pub layers: Vec<BasicLayer>,
    pub norms: Vec<LayerNorm>,
    config: SwinConfig,
}

impl SwinTransformer {
    pub fn new(config: SwinConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            config.patch_size,
            config.in_channels,
            config.embed_dim,
            true,
            vb.pp("patch_embed"),
        )?;

        let num_layers = config.depths.len();
        let mut layers = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let dim = config.embed_dim * (1 << i);
            let downsample = i < num_layers - 1;

            let layer = BasicLayer::new(
                dim,
                config.depths[i],
                config.num_heads[i],
                config.window_size,
                config.mlp_ratio,
                downsample,
                vb.pp(format!("layers.{}", i)),
            )?;
            layers.push(layer);

            let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp(format!("norm{}", i)))?;
            norms.push(norm);
        }

        Ok(Self {
            patch_embed,
            layers,
            norms,
            config,
        })
    }

    /// Forward pass returning features at each stage
    /// Returns: (x1, x2, x3, x4) where each xi has shape [B, Ci, Hi, Wi]
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let x = self.patch_embed.forward(x)?;

        let (_, _, wh, ww) = x.dims4()?;

        // Flatten and transpose: [B, C, H, W] -> [B, H*W, C]
        let mut x = x.flatten_from(2)?.transpose(1, 2)?;

        let mut h = wh;
        let mut w = ww;
        let mut outs = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let (x_out, out_h, out_w, x_down, new_h, new_w) = layer.forward(&x, h, w)?;

            // Apply norm and reshape to spatial
            let x_normed = self.norms[i].forward(&x_out)?;
            let c = self.config.embed_dim * (1 << i);
            let out = x_normed
                .reshape((x_normed.dim(0)?, out_h, out_w, c))?
                .permute((0, 3, 1, 2))?;
            outs.push(out);

            x = x_down;
            h = new_h;
            w = new_w;
        }

        Ok(outs)
    }
}

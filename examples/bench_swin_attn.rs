//! Benchmark Swin attention parts

use candle_core::{Device, Tensor, D, Module};
use candle_nn::Linear;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    // Swin-L config at 1024x1024 input
    // After patch_embed: 256x256, window_size=12
    // H=256, W=256, ws=12 => ceil(256/12)=22 windows per side (padded to 264)
    // num_windows = 22*22 = 484 per batch
    // b_ = batch * num_windows = 1 * 484 = 484
    // n = window_size^2 = 12*12 = 144
    let b_ = 484;
    let n = 144; // window_size^2 = 12*12
    let c = 192; // channels in first stage
    let num_heads = 6;
    let head_dim = c / num_heads;

    println!("Testing with b_={}, n={}, c={}, heads={}", b_, n, c, num_heads);

    // Create test tensors
    let x = Tensor::randn(0f32, 1.0, (b_, n, c), &device)?;

    // Create linear layers like candle does
    let qkv_weight = Tensor::randn(0f32, 0.02, (3 * c, c), &device)?;
    let qkv_bias = Tensor::zeros((3 * c,), candle_core::DType::F32, &device)?;
    let qkv = Linear::new(qkv_weight, Some(qkv_bias));

    // Warmup
    let _ = qkv.forward(&x)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }

    // QKV projection
    let start = Instant::now();
    let qkv_out = qkv.forward(&x)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("QKV projection: {:?}", start.elapsed());

    // Reshape/permute
    let start = Instant::now();
    let qkv_r = qkv_out.reshape((b_, n, 3, num_heads, head_dim))?;
    let qkv_p = qkv_r.permute((2, 0, 3, 1, 4))?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Reshape/permute: {:?}", start.elapsed());

    // Get Q, K, V
    let q = qkv_p.get(0)?;
    let k = qkv_p.get(1)?;
    let v = qkv_p.get(2)?;

    // Q @ K^T
    let start = Instant::now();
    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let attn = q.matmul(&k_t)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Q @ K^T: {:?}", start.elapsed());

    // Softmax
    let start = Instant::now();
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Softmax: {:?}", start.elapsed());

    // Attn @ V
    let start = Instant::now();
    let out = attn.matmul(&v)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Attn @ V: {:?}", start.elapsed());

    // Output reshape
    let start = Instant::now();
    let out = out.transpose(1, 2)?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Output transpose: {:?}", start.elapsed());

    let start = Instant::now();
    let out = out.contiguous()?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Output contiguous: {:?}", start.elapsed());

    let start = Instant::now();
    let _out = out.reshape((b_, n, c))?;
    if let Device::Metal(m) = &device { m.wait_until_completed()?; }
    println!("Output final reshape: {:?}", start.elapsed());

    Ok(())
}

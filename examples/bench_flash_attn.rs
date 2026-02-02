//! Benchmark flash attention vs standard attention

use candle_core::{Device, Tensor, D};
use std::time::Instant;

#[cfg(feature = "flash-attn")]
use candle_mps_flash_attention::flash_attention;

fn sync_metal(t: &Tensor) -> anyhow::Result<()> {
    // Force sync by reading a value
    let _ = t.flatten_all()?.get(0)?.to_scalar::<f32>()?;
    Ok(())
}

fn bench_config(device: &Device, batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> anyhow::Result<()> {
    println!("\n=== batch={}, seq={}, heads={}, head_dim={} ===", batch, seq_len, num_heads, head_dim);

    // Create Q, K, V in shape [batch, num_heads, seq, head_dim]
    let q = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), device)?;
    let k = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), device)?;
    let v = Tensor::randn(0f32, 1.0, (batch, num_heads, seq_len, head_dim), device)?;

    // Warmup standard attention
    let _ = standard_attention(&q, &k, &v)?;
    sync_metal(&q)?;

    // Benchmark standard attention
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = standard_attention(&q, &k, &v)?;
    }
    sync_metal(&q)?;
    println!("Standard attention: {:.1}ms/call", start.elapsed().as_millis() as f64 / iters as f64);

    // Flash attention (if available)
    #[cfg(feature = "flash-attn")]
    {
        // Flash attention expects [batch, seq, heads, head_dim]
        let q_fa = q.transpose(1, 2)?.contiguous()?;
        let k_fa = k.transpose(1, 2)?.contiguous()?;
        let v_fa = v.transpose(1, 2)?.contiguous()?;

        // Warmup
        let _ = flash_attention(&q_fa, &k_fa, &v_fa, false)?;
        sync_metal(&q)?;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = flash_attention(&q_fa, &k_fa, &v_fa, false)?;
        }
        sync_metal(&q)?;
        println!("Flash attention: {:.1}ms/call", start.elapsed().as_millis() as f64 / iters as f64);
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    // Swin-L config: b_=484, n=144, heads=6, head_dim=32
    // This has 484 batches which causes 484 dispatch loops in MFA!
    let b_ = 484;
    let n = 144;
    let num_heads = 6;
    let head_dim = 32;

    // Test Swin config (many small batches) - MFA is BAD here
    bench_config(&device, b_, n, num_heads, head_dim)?;

    // Compare: same total elements but batch=1
    // batch=1, seq=484*144=69696 would be too big, so approximate
    bench_config(&device, 1, 144, 484 * 6, head_dim)?; // Move batch into heads

    // Test typical LLM configs (single batch, long sequences) - MFA should be better here
    bench_config(&device, 1, 512, 32, 128)?;
    bench_config(&device, 1, 1024, 32, 128)?;
    bench_config(&device, 1, 2048, 32, 128)?;
    bench_config(&device, 1, 4096, 32, 128)?;
    bench_config(&device, 1, 8192, 32, 128)?;

    // Test small batch long sequence
    bench_config(&device, 4, 2048, 8, 64)?;

    Ok(())
}

fn standard_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let q = (q * scale)?;
    let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let attn = candle_nn::ops::softmax_last_dim(&attn)?;
    attn.matmul(v)
}

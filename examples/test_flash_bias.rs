//! Test flash attention with bias vs standard attention with bias

use candle_core::{Device, Tensor, D};

#[cfg(feature = "flash-attn")]
use candle_mps_flash_attention::{flash_attention_with_bias, flash_attention_with_repeating_bias};

fn test_simple_bias(device: &Device) -> anyhow::Result<()> {
    println!("\n========== TEST: Simple Bias ==========");

    // Simple test case: batch=2, heads=4, seq=16, head_dim=32
    let batch = 2;
    let num_heads = 4;
    let seq_len = 16;
    let head_dim = 32;

    println!("Test config: batch={}, heads={}, seq={}, head_dim={}", batch, num_heads, seq_len, head_dim);

    // Create Q, K, V in shape [batch, num_heads, seq, head_dim]
    let q = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), device)?;
    let k = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), device)?;
    let v = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), device)?;

    // Create bias: [1, num_heads, seq, seq] - broadcast across batch
    let bias = Tensor::randn(0f32, 0.5, (1, num_heads, seq_len, seq_len), device)?;

    println!("\nQ shape: {:?}", q.shape());
    println!("bias shape: {:?}", bias.shape());

    // Standard attention with bias
    let scale = 1.0 / (head_dim as f64).sqrt();
    let q_scaled = (&q * scale)?;
    let attn_scores = q_scaled.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let attn_with_bias = attn_scores.broadcast_add(&bias)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_with_bias)?;
    let standard_out = attn_probs.matmul(&v)?;

    // Force sync
    let _ = standard_out.sum_all()?.to_scalar::<f32>()?;

    // Flash attention with bias
    #[cfg(feature = "flash-attn")]
    {
        let inv_scale = 1.0 / scale;
        let scaled_bias = (&bias * inv_scale)?;

        let flash_out = flash_attention_with_bias(&q, &k, &v, &scaled_bias, false)?;
        let _ = flash_out.sum_all()?.to_scalar::<f32>()?;

        // Compare outputs
        let diff = (&standard_out - &flash_out)?.abs()?;
        let max_diff_val: f32 = diff.max_all()?.to_scalar()?;
        let flash_sum: f32 = flash_out.abs()?.sum_all()?.to_scalar()?;
        let std_sum: f32 = standard_out.abs()?.sum_all()?.to_scalar()?;

        println!("Sum of abs(standard): {:.4}", std_sum);
        println!("Sum of abs(flash):    {:.4}", flash_sum);
        println!("Max diff: {:.6}", max_diff_val);

        if flash_sum < 0.0001 {
            println!("*** FAIL: FLASH OUTPUT IS ALL ZEROS! ***");
        } else if max_diff_val < 0.01 {
            println!("*** PASS ***");
        } else {
            println!("*** FAIL: OUTPUTS DIFFER ***");
        }
    }

    Ok(())
}

fn test_repeating_bias(device: &Device) -> anyhow::Result<()> {
    println!("\n========== TEST: Repeating Bias (Swin-like) ==========");

    // Swin-like config:
    // batch = num_windows * batch_size = 4 windows * 2 images = 8
    // The bias pattern repeats every num_windows (4) batches
    let num_windows = 4;
    let batch_per_window = 2;
    let batch = num_windows * batch_per_window; // 8 total
    let num_heads = 4;
    let seq_len = 49; // 7x7 window
    let head_dim = 32;

    println!("Config: batch={}, heads={}, seq={}, head_dim={}", batch, num_heads, seq_len, head_dim);
    println!("  num_windows={}, batch_per_window={}", num_windows, batch_per_window);

    // Create Q, K, V
    let q = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), device)?;
    let k = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), device)?;
    let v = Tensor::randn(0f32, 0.1, (batch, num_heads, seq_len, head_dim), device)?;

    // Repeating bias: [num_windows, num_heads, seq, seq]
    // This is the typical Swin case: position bias repeats for each window pattern
    let bias = Tensor::randn(0f32, 0.5, (num_windows, num_heads, seq_len, seq_len), device)?;

    println!("Q shape: {:?}", q.shape());
    println!("bias shape: {:?}", bias.shape());

    // Standard attention - manually expand bias for each batch
    let scale = 1.0 / (head_dim as f64).sqrt();
    let q_scaled = (&q * scale)?;
    let attn_scores = q_scaled.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

    // Expand bias: for batch b, use bias[b % num_windows]
    // Shape [batch, num_heads, seq, seq]
    let mut bias_expanded_vec: Vec<Tensor> = Vec::new();
    for b in 0..batch {
        let pattern_idx = b % num_windows;
        let bias_slice = bias.get(pattern_idx)?; // [num_heads, seq, seq]
        bias_expanded_vec.push(bias_slice.unsqueeze(0)?);
    }
    let bias_expanded = Tensor::cat(&bias_expanded_vec, 0)?; // [batch, num_heads, seq, seq]

    let attn_with_bias = attn_scores.broadcast_add(&bias_expanded)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_with_bias)?;
    let standard_out = attn_probs.matmul(&v)?;
    let _ = standard_out.sum_all()?.to_scalar::<f32>()?;

    // Flash attention with repeating bias
    #[cfg(feature = "flash-attn")]
    {
        let inv_scale = 1.0 / scale;
        let scaled_bias = (&bias * inv_scale)?;

        let flash_out = flash_attention_with_repeating_bias(&q, &k, &v, &scaled_bias, num_windows, false)?;
        let _ = flash_out.sum_all()?.to_scalar::<f32>()?;

        // Compare outputs
        let diff = (&standard_out - &flash_out)?.abs()?;
        let max_diff_val: f32 = diff.max_all()?.to_scalar()?;
        let flash_sum: f32 = flash_out.abs()?.sum_all()?.to_scalar()?;
        let std_sum: f32 = standard_out.abs()?.sum_all()?.to_scalar()?;

        println!("Sum of abs(standard): {:.4}", std_sum);
        println!("Sum of abs(flash):    {:.4}", flash_sum);
        println!("Max diff: {:.6}", max_diff_val);

        if flash_sum < 0.0001 {
            println!("*** FAIL: FLASH OUTPUT IS ALL ZEROS! ***");
        } else if max_diff_val < 0.01 {
            println!("*** PASS ***");
        } else {
            println!("*** FAIL: OUTPUTS DIFFER ***");
        }
    }

    Ok(())
}

fn test_swin_config(device: &Device) -> anyhow::Result<()> {
    println!("\n========== TEST: Actual Swin-L Config ==========");

    // Swin-L at 1024x1024: 484 windows, 144 tokens per window
    let b_ = 484;
    let n = 144;
    let num_heads = 6;
    let head_dim = 32;
    let _c = num_heads * head_dim;

    println!("Config: b_={}, n={}, heads={}, head_dim={}", b_, n, num_heads, head_dim);

    // Create Q, K, V
    let q = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;
    let k = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;
    let v = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;

    // Simple bias [1, num_heads, n, n] - broadcast across all batches (non-shifted case)
    let bias = Tensor::randn(0f32, 0.5, (1, num_heads, n, n), device)?;

    println!("Q shape: {:?}", q.shape());
    println!("bias shape: {:?}", bias.shape());

    // Standard attention with bias
    let scale = 1.0 / (head_dim as f64).sqrt();
    let q_scaled = (&q * scale)?;
    let attn_scores = q_scaled.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
    let attn_with_bias = attn_scores.broadcast_add(&bias)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_with_bias)?;
    let standard_out = attn_probs.matmul(&v)?;
    let _ = standard_out.sum_all()?.to_scalar::<f32>()?;
    let std_sum: f32 = standard_out.abs()?.sum_all()?.to_scalar()?;
    println!("Sum of abs(standard): {:.4}", std_sum);

    // Flash attention with bias
    #[cfg(feature = "flash-attn")]
    {
        let inv_scale = 1.0 / scale;
        let scaled_bias = (&bias * inv_scale)?;

        let flash_out = flash_attention_with_bias(&q, &k, &v, &scaled_bias, false)?;
        let _ = flash_out.sum_all()?.to_scalar::<f32>()?;

        let diff = (&standard_out - &flash_out)?.abs()?;
        let max_diff_val: f32 = diff.max_all()?.to_scalar()?;
        let flash_sum: f32 = flash_out.abs()?.sum_all()?.to_scalar()?;

        println!("Sum of abs(flash):    {:.4}", flash_sum);
        println!("Max diff: {:.6}", max_diff_val);

        if flash_sum < 0.0001 {
            println!("*** FAIL: FLASH OUTPUT IS ALL ZEROS! ***");
        } else if max_diff_val < 0.1 {
            println!("*** PASS ***");
        } else {
            println!("*** FAIL: OUTPUTS DIFFER ***");
        }
    }

    Ok(())
}

fn test_swin_shifted_window(device: &Device) -> anyhow::Result<()> {
    println!("\n========== TEST: Swin Shifted Window (repeating bias) ==========");

    // Swin-L at 1024x1024: 484 windows, but shifted means n_windows unique patterns
    // For simplicity, assume n_windows = 4 (2x2 shift patterns) and 121 batches per pattern
    let n_windows = 484; // Each window is a unique pattern in shifted case
    let batch_per_pattern = 1;
    let b_ = n_windows * batch_per_pattern;
    let n = 144;
    let num_heads = 6;
    let head_dim = 32;

    println!("Config: b_={}, n={}, heads={}, head_dim={}", b_, n, num_heads, head_dim);
    println!("  n_windows={}, batch_per_pattern={}", n_windows, batch_per_pattern);

    // Create Q, K, V
    let q = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;
    let k = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;
    let v = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;

    // Repeating bias: [n_windows, num_heads, n, n]
    let bias = Tensor::randn(0f32, 0.5, (n_windows, num_heads, n, n), device)?;

    println!("Q shape: {:?}", q.shape());
    println!("bias shape: {:?}", bias.shape());

    // Standard attention - expand bias for each batch
    let scale = 1.0 / (head_dim as f64).sqrt();
    let q_scaled = (&q * scale)?;
    let attn_scores = q_scaled.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

    // For repeating bias with n_windows == b_, each batch uses its own bias
    // bias[b % n_windows] = bias[b] when n_windows == b_
    let bias_expanded = bias.clone(); // [b_, num_heads, n, n]

    let attn_with_bias = attn_scores.broadcast_add(&bias_expanded)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_with_bias)?;
    let standard_out = attn_probs.matmul(&v)?;
    let _ = standard_out.sum_all()?.to_scalar::<f32>()?;
    let std_sum: f32 = standard_out.abs()?.sum_all()?.to_scalar()?;
    println!("Sum of abs(standard): {:.4}", std_sum);

    // Flash attention with repeating bias
    #[cfg(feature = "flash-attn")]
    {
        let inv_scale = 1.0 / scale;
        let scaled_bias = (&bias * inv_scale)?;

        let flash_out = flash_attention_with_repeating_bias(&q, &k, &v, &scaled_bias, n_windows, false)?;
        let _ = flash_out.sum_all()?.to_scalar::<f32>()?;

        let diff = (&standard_out - &flash_out)?.abs()?;
        let max_diff_val: f32 = diff.max_all()?.to_scalar()?;
        let flash_sum: f32 = flash_out.abs()?.sum_all()?.to_scalar()?;

        println!("Sum of abs(flash):    {:.4}", flash_sum);
        println!("Max diff: {:.6}", max_diff_val);

        if flash_sum < 0.0001 {
            println!("*** FAIL: FLASH OUTPUT IS ALL ZEROS! ***");
        } else if max_diff_val < 0.1 {
            println!("*** PASS ***");
        } else {
            println!("*** FAIL: OUTPUTS DIFFER ***");
        }
    }

    Ok(())
}

fn test_birefnet_exact_config(device: &Device) -> anyhow::Result<()> {
    println!("\n========== TEST: BiRefNet Exact Config (shifted) ==========");

    // Exact config from BiRefNet debug output:
    // q=[484, 6, 144, 32] combined=[484, 6, 144, 144] n_windows=484
    let b_ = 484;
    let num_heads = 6;
    let n = 144;
    let head_dim = 32;
    let n_windows = 484;

    println!("Config: b_={}, heads={}, n={}, head_dim={}, n_windows={}", b_, num_heads, n, head_dim, n_windows);

    // Create Q, K, V
    let q = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;
    let k = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;
    let v = Tensor::randn(0f32, 0.1, (b_, num_heads, n, head_dim), device)?;

    // Combined bias (position bias + window mask): [n_windows, num_heads, n, n]
    let bias = Tensor::randn(0f32, 0.5, (n_windows, num_heads, n, n), device)?;

    println!("Q shape: {:?}", q.shape());
    println!("bias shape: {:?}", bias.shape());

    // Standard attention
    let scale = 1.0 / (head_dim as f64).sqrt();
    let q_scaled = (&q * scale)?;
    let attn_scores = q_scaled.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

    // Since n_windows == b_, each batch b uses bias[b]
    let attn_with_bias = attn_scores.broadcast_add(&bias)?;
    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_with_bias)?;
    let standard_out = attn_probs.matmul(&v)?;
    let _ = standard_out.sum_all()?.to_scalar::<f32>()?;
    let std_sum: f32 = standard_out.abs()?.sum_all()?.to_scalar()?;
    println!("Sum of abs(standard): {:.4}", std_sum);

    // Flash attention with repeating bias
    #[cfg(feature = "flash-attn")]
    {
        let inv_scale = 1.0 / scale;
        let scaled_bias = (&bias * inv_scale)?;

        let flash_out = flash_attention_with_repeating_bias(&q, &k, &v, &scaled_bias, n_windows, false)?;
        let _ = flash_out.sum_all()?.to_scalar::<f32>()?;

        let diff = (&standard_out - &flash_out)?.abs()?;
        let max_diff_val: f32 = diff.max_all()?.to_scalar()?;
        let flash_sum: f32 = flash_out.abs()?.sum_all()?.to_scalar()?;

        println!("Sum of abs(flash):    {:.4}", flash_sum);
        println!("Max diff: {:.6}", max_diff_val);

        // Also check a few specific values
        let std_flat: Vec<f32> = standard_out.flatten_all()?.narrow(0, 0, 10)?.to_vec1()?;
        let flash_flat: Vec<f32> = flash_out.flatten_all()?.narrow(0, 0, 10)?.to_vec1()?;
        println!("Standard first 5: {:?}", &std_flat[..5]);
        println!("Flash first 5:    {:?}", &flash_flat[..5]);

        if flash_sum < 0.0001 {
            println!("*** FAIL: FLASH OUTPUT IS ALL ZEROS! ***");
        } else if max_diff_val < 0.1 {
            println!("*** PASS ***");
        } else {
            println!("*** FAIL: OUTPUTS DIFFER ***");
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    test_simple_bias(&device)?;
    test_repeating_bias(&device)?;
    test_swin_config(&device)?;
    test_swin_shifted_window(&device)?;
    test_birefnet_exact_config(&device)?;

    Ok(())
}

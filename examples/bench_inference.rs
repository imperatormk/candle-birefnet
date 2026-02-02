//! Benchmark BiRefNet inference stages

use candle_core::{Device, DType, Tensor, Module};
use candle_nn::VarBuilder;
use candle_birefnet::{BiRefNet, birefnet::BiRefNetConfig};
use std::collections::HashMap;
use std::time::Instant;

fn sync(t: &Tensor) {
    // Force GPU sync by reading a value back to CPU
    let _ = t.sum_all().and_then(|s| s.to_scalar::<f32>());
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    // Load model
    let weights_path = std::env::var("HOME")?
        + "/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet/snapshots/26e7919b869d089e5096e898c0492898f935604c/model.safetensors";

    println!("Loading model...");
    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let remapped: HashMap<String, Tensor> = tensors.into_iter().collect();
    let config = BiRefNetConfig::swin_l();
    let vb = VarBuilder::from_tensors(remapped, dtype, &device);
    let model = BiRefNet::new(config.clone(), vb)?;

    // Create random input
    let x = Tensor::randn(0f32, 1.0, (1, 3, 1024, 1024), &device)?;

    // Warmup
    println!("\nWarmup...");
    let warmup_out = model.backbone.forward(&x)?;
    sync(&warmup_out[0]);

    // Benchmark backbone
    println!("\n=== Benchmarking backbone (single pass) ===");
    let start = Instant::now();
    let features = model.backbone.forward(&x)?;
    sync(&features[0]);
    println!("Backbone: {:?}", start.elapsed());

    // Benchmark half-scale backbone
    let x_half = x.upsample_bilinear2d(512, 512, true)?;
    let start = Instant::now();
    let features_half = model.backbone.forward(&x_half)?;
    sync(&features_half[0]);
    println!("Backbone (half scale): {:?}", start.elapsed());

    // Benchmark squeeze module
    let x1 = &features[0];
    let x2 = &features[1];
    let x3 = &features[2];
    let x4 = &features[3];

    // Build x4_cxt like in forward_logits
    let (_, _, h4, w4) = x4.dims4()?;
    let x1_to_x4 = x1.upsample_bilinear2d(h4, w4, true)?;
    let x2_to_x4 = x2.upsample_bilinear2d(h4, w4, true)?;
    let x3_to_x4 = x3.upsample_bilinear2d(h4, w4, true)?;

    // With mul_scl_ipt, channels are doubled
    let x1_cat = Tensor::cat(&[x1, x1], 1)?;  // Simulate doubled channels
    let x2_cat = Tensor::cat(&[x2, x2], 1)?;
    let x3_cat = Tensor::cat(&[x3, x3], 1)?;
    let x4_cat = Tensor::cat(&[x4, x4], 1)?;

    let x1_to_x4_cat = Tensor::cat(&[&x1_to_x4, &x1_to_x4], 1)?;
    let x2_to_x4_cat = Tensor::cat(&[&x2_to_x4, &x2_to_x4], 1)?;
    let x3_to_x4_cat = Tensor::cat(&[&x3_to_x4, &x3_to_x4], 1)?;

    let x4_cxt = Tensor::cat(&[&x1_to_x4_cat, &x2_to_x4_cat, &x3_to_x4_cat, &x4_cat], 1)?;
    println!("x4_cxt shape: {:?}", x4_cxt.dims());

    let start = Instant::now();
    let x4_squeezed = model.squeeze_module.forward(&x4_cxt)?;
    sync(&x4_squeezed);
    println!("Squeeze module: {:?}", start.elapsed());

    // Benchmark decoder
    let start = Instant::now();
    let decoder_out = model.decoder.forward(&x, &x1_cat, &x2_cat, &x3_cat, &x4_squeezed)?;
    sync(&decoder_out);
    println!("Decoder: {:?}", start.elapsed());

    // Full inference
    println!("\n=== Full inference ===");
    let start = Instant::now();
    let logits = model.forward_logits(&x)?;
    sync(&logits);
    println!("Total: {:?}", start.elapsed());

    Ok(())
}

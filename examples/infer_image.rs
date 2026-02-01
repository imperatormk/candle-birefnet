//! Run BiRefNet inference on an image

use candle_core::{Device, DType, Tensor, Module};
use candle_nn::VarBuilder;
use candle_birefnet::{BiRefNet, birefnet::BiRefNetConfig};
use image::{GenericImageView, ImageBuffer, Luma};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path> [output_path]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("output_mask.png");

    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    // Load weights
    let weights_path = std::env::var("HOME")?
        + "/.cache/huggingface/hub/models--ZhengPeng7--BiRefNet/snapshots/26e7919b869d089e5096e898c0492898f935604c/model.safetensors";

    println!("Loading model...");
    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let remapped: HashMap<String, Tensor> = tensors.into_iter().collect();

    let config = BiRefNetConfig::swin_l();
    let vb = VarBuilder::from_tensors(remapped, dtype, &device);
    let model = BiRefNet::new(config, vb)?;

    // Load and preprocess image
    println!("Loading image: {}", input_path);
    let img = image::open(input_path)?;
    let (orig_w, orig_h) = img.dimensions();
    println!("Original size: {}x{}", orig_w, orig_h);

    // Resize to 1024x1024 for inference (BiRefNet default)
    // Use Triangle (bilinear) to match torchvision.transforms.Resize default
    let resized = img.resize_exact(1024, 1024, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    // Convert to tensor [1, 3, 1024, 1024], with ImageNet normalization
    // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let mut data = vec![0f32; 3 * 1024 * 1024];
    for y in 0..1024 {
        for x in 0..1024 {
            let pixel = rgb.get_pixel(x, y);
            let idx = (y * 1024 + x) as usize;
            data[idx] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];                    // R
            data[1024 * 1024 + idx] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];      // G
            data[2 * 1024 * 1024 + idx] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];  // B
        }
    }

    let input = Tensor::from_vec(data, (1, 3, 1024, 1024), &device)?;

    // Run inference
    println!("Running inference...");
    let start = std::time::Instant::now();
    let logits = model.forward_logits(&input)?;
    // Force sync by reading a value
    let _ = logits.sum_all()?.to_scalar::<f32>()?;
    let elapsed = start.elapsed();
    println!("Inference time: {:?}", elapsed);

    // Debug: check output before sigmoid
    let out_flat_raw = logits.flatten_all()?.to_vec1::<f32>()?;
    let raw_min = out_flat_raw.iter().cloned().fold(f32::INFINITY, f32::min);
    let raw_max = out_flat_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let raw_sum: f32 = out_flat_raw.iter().sum();
    println!("Raw output (before sigmoid): min={:.4}, max={:.4}, sum={:.4}", raw_min, raw_max, raw_sum);

    // Apply sigmoid
    let output = candle_nn::ops::sigmoid(&logits)?;

    // Get output stats
    let out_flat = output.flatten_all()?;
    let min_val = out_flat.min(0)?.to_scalar::<f32>()?;
    let max_val = out_flat.max(0)?.to_scalar::<f32>()?;
    println!("Output range: [{:.4}, {:.4}]", min_val, max_val);

    // Convert output to image
    let mask_data: Vec<f32> = output.squeeze(0)?.squeeze(0)?.to_vec2()?
        .into_iter().flatten().collect();

    // Create grayscale mask at 1024x1024
    let mask_img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(1024, 1024, |x, y| {
        let idx = (y * 1024 + x) as usize;
        let v = (mask_data[idx] * 255.0).clamp(0.0, 255.0) as u8;
        Luma([v])
    });

    // Resize back to original size
    let mask_resized = image::imageops::resize(
        &mask_img,
        orig_w,
        orig_h,
        image::imageops::FilterType::Lanczos3,
    );

    // Save
    mask_resized.save(output_path)?;
    println!("Saved mask to: {}", output_path);

    Ok(())
}

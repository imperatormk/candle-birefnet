//! Test full BiRefNet model (backbone + decoder)

use candle_core::{Device, DType, Tensor, Module};
use candle_nn::VarMap;
use candle_birefnet::{BiRefNet, birefnet::BiRefNetConfig};

fn main() -> candle_core::Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    println!("Testing full BiRefNet model...\n");

    // Use swin_t config for faster testing
    let config = BiRefNetConfig::swin_t();
    println!("Config: swin_t backbone");
    println!("  lateral_channels: {:?}", config.lateral_channels);
    println!("  use_aspp_deformable: {}", config.use_aspp_deformable);

    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

    println!("\nBuilding model...");
    let model = BiRefNet::new(config, vb)?;

    // Test input: 256x256 image (smaller for faster test)
    let batch = 1;
    let h = 256;
    let w = 256;
    println!("Input shape: [{}, 3, {}, {}]", batch, h, w);

    let input = Tensor::randn(0.0f32, 1.0, (batch, 3, h, w), &device)?;

    // Warmup
    println!("\nWarmup...");
    for _ in 0..2 {
        let _ = model.forward(&input)?;
    }
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }

    // Benchmark
    println!("Running inference...");
    let n_iters = 3;
    let start = std::time::Instant::now();
    let mut output = None;
    for _ in 0..n_iters {
        output = Some(model.forward(&input)?);
    }
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }
    let elapsed = start.elapsed();

    let output = output.unwrap();
    println!("\nOutput shape: {:?}", output.dims());
    println!("Expected: [{}, 1, {}, {}] (segmentation mask)", batch, h, w);

    println!("\nAvg inference time: {:?}", elapsed / n_iters as u32);

    // Check output stats
    let out_flat = output.flatten_all()?;
    let min_val = out_flat.min(0)?.to_scalar::<f32>()?;
    let max_val = out_flat.max(0)?.to_scalar::<f32>()?;
    println!("Output range: [{:.4}, {:.4}]", min_val, max_val);

    println!("\nBiRefNet test passed!");

    Ok(())
}

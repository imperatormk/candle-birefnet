//! Test the BiRefNet decoder with random inputs

use candle_core::{Device, DType, Tensor};
use candle_nn::VarMap;
use candle_birefnet::birefnet::{BiRefNet, BiRefNetConfig};

fn main() -> candle_core::Result<()> {
    // Use Metal on macOS, CPU otherwise
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let dtype = DType::F32;

    println!("Creating BiRefNet decoder...");

    // Create config
    let config = BiRefNetConfig::default();
    println!("  Config: {:?}x{:?}, backbone={}", config.size.0, config.size.1, config.backbone);
    println!("  Lateral channels: {:?}", config.lateral_channels);

    // Create varmap with random weights
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

    // Initialize model (this creates all the weights)
    let model = BiRefNet::new(config.clone(), vb)?;
    println!("  Model created successfully!");

    // Create dummy inputs matching expected shapes
    // x: original image features [B, 3, H, W]
    // x1-x4: encoder features at different scales
    let batch = 1;
    // Use smaller size for quick test
    let (h, w) = (256, 256);

    let x = Tensor::randn(0.0f32, 1.0, (batch, 3, h, w), &device)?;
    let x1 = Tensor::randn(0.0f32, 1.0, (batch, config.lateral_channels[3], h / 4, w / 4), &device)?;
    let x2 = Tensor::randn(0.0f32, 1.0, (batch, config.lateral_channels[2], h / 8, w / 8), &device)?;
    let x3 = Tensor::randn(0.0f32, 1.0, (batch, config.lateral_channels[1], h / 16, w / 16), &device)?;
    let x4 = Tensor::randn(0.0f32, 1.0, (batch, config.lateral_channels[0], h / 32, w / 32), &device)?;

    println!("\nInput shapes:");
    println!("  x:  {:?}", x.dims());
    println!("  x1: {:?}", x1.dims());
    println!("  x2: {:?}", x2.dims());
    println!("  x3: {:?}", x3.dims());
    println!("  x4: {:?}", x4.dims());

    // Run forward pass
    println!("\nRunning forward pass...");
    let output = model.forward_with_features(&x, &x1, &x2, &x3, &x4)?;
    println!("  Output shape: {:?}", output.dims());

    // Check output stats
    let mean = output.mean_all()?.to_scalar::<f32>()?;
    let max = output.max_all()?.to_scalar::<f32>()?;
    let min = output.min_all()?.to_scalar::<f32>()?;
    println!("  Output stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);

    println!("\nDecoder test passed!");
    Ok(())
}

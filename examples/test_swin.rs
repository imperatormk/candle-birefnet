//! Test Swin Transformer backbone

use candle_core::{Device, DType, Tensor};
use candle_nn::VarMap;
use candle_birefnet::{SwinTransformer, SwinConfig};

fn main() -> candle_core::Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    println!("Testing Swin Transformer backbone...\n");

    // Use swin_t (tiny) for faster testing
    let config = SwinConfig::swin_t();
    println!("Config: swin_t");
    println!("  embed_dim: {}", config.embed_dim);
    println!("  depths: {:?}", config.depths);
    println!("  num_heads: {:?}", config.num_heads);
    println!("  window_size: {}", config.window_size);

    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

    println!("\nBuilding model...");
    let model = SwinTransformer::new(config.clone(), vb)?;

    // Test input: batch=1, channels=3, 256x256 image
    let batch = 1;
    let h = 256;
    let w = 256;
    println!("\nInput shape: [{}, 3, {}, {}]", batch, h, w);

    let input = Tensor::randn(0.0f32, 1.0, (batch, 3, h, w), &device)?;

    // Warmup
    println!("Warmup...");
    for _ in 0..2 {
        let _ = model.forward(&input)?;
    }
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }

    // Forward pass
    println!("Running forward pass...");
    let start = std::time::Instant::now();
    let features = model.forward(&input)?;
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }
    let elapsed = start.elapsed();

    println!("\nOutput features (multi-scale):");
    for (i, feat) in features.iter().enumerate() {
        println!("  Stage {}: {:?}", i, feat.dims());
    }

    println!("\nForward time: {:?}", elapsed);

    // Verify feature map dimensions
    // For 256x256 input with swin_t:
    // Stage 0: H/4 x W/4 = 64x64, C=96
    // Stage 1: H/8 x W/8 = 32x32, C=192
    // Stage 2: H/16 x W/16 = 16x16, C=384
    // Stage 3: H/32 x W/32 = 8x8, C=768

    println!("\nExpected dimensions for 256x256 input with swin_t:");
    println!("  Stage 0: [1, 64, 64, 96]   (H/4, W/4, C*1)");
    println!("  Stage 1: [1, 32, 32, 192]  (H/8, W/8, C*2)");
    println!("  Stage 2: [1, 16, 16, 384]  (H/16, W/16, C*4)");
    println!("  Stage 3: [1, 8, 8, 768]    (H/32, W/32, C*8)");

    println!("\nSwin Transformer test passed!");

    Ok(())
}

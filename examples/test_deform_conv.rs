//! Test deformable convolution with Metal kernel

use candle_core::{Device, DType, Tensor};
use candle_nn::VarMap;
use candle_birefnet::DeformableConv2d;

fn main() -> candle_core::Result<()> {
    // Use Metal on macOS
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let dtype = DType::F32;

    println!("Testing DeformableConv2d on {:?}", device);
    println!("{}", "=".repeat(50));

    // Create varmap with random weights
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);

    // Create a deformable conv layer
    let in_channels = 64;
    let out_channels = 128;
    let kernel_size = 3;
    let stride = 1;
    let padding = 1;

    println!("\nCreating DeformableConv2d:");
    println!("  in_channels:  {}", in_channels);
    println!("  out_channels: {}", out_channels);
    println!("  kernel_size:  {}", kernel_size);
    println!("  stride:       {}", stride);
    println!("  padding:      {}", padding);

    let deform_conv = DeformableConv2d::new(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        vb,
    )?;

    // Create test input
    let batch = 1;
    let h = 32;
    let w = 32;
    let input = Tensor::randn(0.0f32, 1.0, (batch, in_channels, h, w), &device)?;
    println!("\nInput shape: {:?}", input.dims());

    // Run forward pass
    println!("\nRunning forward pass...");
    let start = std::time::Instant::now();
    let output = deform_conv.forward(&input)?;

    // Sync Metal if needed
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }

    let elapsed = start.elapsed();
    println!("  Forward pass took: {:?}", elapsed);
    println!("  Output shape: {:?}", output.dims());

    // Check output stats
    let mean = output.mean_all()?.to_scalar::<f32>()?;
    let max = output.max_all()?.to_scalar::<f32>()?;
    let min = output.min_all()?.to_scalar::<f32>()?;
    println!("  Output stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);

    // Verify output dimensions
    let expected_h = (h + 2 * padding - kernel_size) / stride + 1;
    let expected_w = (w + 2 * padding - kernel_size) / stride + 1;
    let (ob, oc, oh, ow) = output.dims4()?;

    assert_eq!(ob, batch, "Batch size mismatch");
    assert_eq!(oc, out_channels, "Output channels mismatch");
    assert_eq!(oh, expected_h, "Output height mismatch");
    assert_eq!(ow, expected_w, "Output width mismatch");

    println!("\n[PASS] DeformableConv2d test passed!");
    println!("  Expected output: [{}, {}, {}, {}]", batch, out_channels, expected_h, expected_w);
    println!("  Actual output:   [{}, {}, {}, {}]", ob, oc, oh, ow);

    // Benchmark with larger input
    println!("\n{}", "=".repeat(50));
    println!("Benchmarking with larger input...");

    let h = 128;
    let w = 128;
    let input = Tensor::randn(0.0f32, 1.0, (batch, in_channels, h, w), &device)?;
    println!("  Input shape: {:?}", input.dims());

    // Warmup
    for _ in 0..3 {
        let _ = deform_conv.forward(&input)?;
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }

    // Benchmark
    let n_iters = 10;
    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = deform_conv.forward(&input)?;
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = &device {
        m.wait_until_completed()?;
    }
    let elapsed = start.elapsed();

    println!("  {} iterations took: {:?}", n_iters, elapsed);
    println!("  Average per iteration: {:?}", elapsed / n_iters as u32);

    Ok(())
}

# candle-birefnet

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) image segmentation model for [candle](https://github.com/huggingface/candle) with Metal GPU acceleration on Apple Silicon.

**No CoreML conversion needed** - runs natively on Metal with full precision!

## Quick Start

```bash
# 1. Build the MFA Swift bridge (required for flash attention)
git clone --recursive https://github.com/mpsops/mps-flash-attention /tmp/mfa
cd /tmp/mfa/swift-bridge && swift build -c release
export MFA_BRIDGE_PATH=/tmp/mfa/swift-bridge/.build/release/libMFABridge.dylib

# 2. Clone and run
git clone https://github.com/imperatormk/candle-birefnet
cd candle-birefnet
cargo run --example infer_image --features flash-attn --release -- photo.jpg mask.png
```

Weights are automatically downloaded from HuggingFace on first run.

## Features

- BiRefNet architecture with Swin Transformer backbone
- Metal GPU acceleration via MPS
- Flash attention for efficient Swin attention
- Deformable convolution via Metal kernels
- Automatic weight download from HuggingFace
- Full precision (no conversion loss!)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
candle-birefnet = { git = "https://github.com/imperatormk/candle-birefnet", features = ["flash-attn"] }
hf-hub = "0.3"  # for downloading weights
```

You also need `libMFABridge.dylib`:

```bash
git clone --recursive https://github.com/mpsops/mps-flash-attention
cd mps-flash-attention/swift-bridge && swift build -c release
export MFA_BRIDGE_PATH=$PWD/.build/release/libMFABridge.dylib
```

## Usage

```rust
use candle_birefnet::{BiRefNet, birefnet::BiRefNetConfig};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    // Download weights from HuggingFace (cached after first run)
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("ZhengPeng7/BiRefNet".to_string());
    let weights_path = repo.get("model.safetensors")?;

    // Load model
    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let vb = VarBuilder::from_tensors(tensors.into_iter().collect(), dtype, &device);
    let config = BiRefNetConfig::swin_l();
    let model = BiRefNet::new(config, vb)?;

    // Prepare input: 1024x1024 RGB image with ImageNet normalization
    // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    let input = load_and_preprocess_image("photo.jpg")?; // (1, 3, 1024, 1024)

    // Run inference
    let logits = model.forward_logits(&input)?;
    let mask = candle_nn::ops::sigmoid(&logits)?; // (1, 1, 1024, 1024)

    // mask values are 0.0-1.0, where 1.0 = foreground
    Ok(())
}
```

### Image Preprocessing

Input images must be:
1. Resized to 1024x1024
2. Normalized with ImageNet mean/std

```rust
use image::GenericImageView;

fn load_and_preprocess_image(path: &str, device: &Device) -> Result<Tensor> {
    let img = image::open(path)?;
    let resized = img.resize_exact(1024, 1024, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let mut data = vec![0f32; 3 * 1024 * 1024];
    for y in 0..1024 {
        for x in 0..1024 {
            let pixel = rgb.get_pixel(x, y);
            let idx = (y * 1024 + x) as usize;
            data[idx] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            data[1024 * 1024 + idx] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            data[2 * 1024 * 1024 + idx] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    Tensor::from_vec(data, (1, 3, 1024, 1024), device)
}
```

### Output Postprocessing

The mask output is 1024x1024. Resize it back to original dimensions:

```rust
use image::{ImageBuffer, Luma};

fn save_mask(mask: &Tensor, orig_w: u32, orig_h: u32, path: &str) -> Result<()> {
    let mask_data: Vec<f32> = mask.squeeze(0)?.squeeze(0)?.to_vec2()?
        .into_iter().flatten().collect();

    let mask_img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(1024, 1024, |x, y| {
        let idx = (y * 1024 + x) as usize;
        let v = (mask_data[idx] * 255.0).clamp(0.0, 255.0) as u8;
        Luma([v])
    });

    let resized = image::imageops::resize(&mask_img, orig_w, orig_h,
        image::imageops::FilterType::Lanczos3);
    resized.save(path)?;
    Ok(())
}
```

## Requirements

- macOS 14+ (Sonoma) or macOS 15+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Rust 1.70+
- `libMFABridge.dylib` from mps-flash-attention

## Credits

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - Original PyTorch implementation
- [candle](https://github.com/huggingface/candle) - Rust ML framework
- [mps-flash-attention](https://github.com/mpsops/mps-flash-attention) - Flash attention for Metal
- [mps-deform-conv](https://github.com/mpsops/mps-deform-conv) - Deformable convolution for Metal
- [candle-mps-flash-attention](https://github.com/mpsops/candle-mps-flash-attention) - Rust bindings for MPS flash attention

## License

MIT

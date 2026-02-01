# candle-birefnet

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) image segmentation model for [candle](https://github.com/huggingface/candle) with Metal GPU acceleration on Apple Silicon.

## Features

- BiRefNet architecture with Swin Transformer backbone
- Metal GPU acceleration via MPS
- Flash attention support for efficient Swin attention (optional)
- Deformable convolution support (optional, requires candle fork)

## Installation

```toml
[dependencies]
candle-birefnet = { git = "https://github.com/mpsops/candle-birefnet" }
```

### With Metal GPU support

```toml
[dependencies]
candle-birefnet = { git = "https://github.com/mpsops/candle-birefnet", features = ["metal"] }
```

### With Flash Attention (fastest)

```toml
[dependencies]
candle-birefnet = { git = "https://github.com/mpsops/candle-birefnet", features = ["flash-attn"] }
```

Requires `libMFABridge.dylib` from [mps-flash-attention](https://github.com/mpsops/mps-flash-attention):

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

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    // Load weights
    let weights = candle_core::safetensors::load("model.safetensors", &device)?;
    let vb = VarBuilder::from_tensors(weights.into_iter().collect(), dtype, &device);

    // Create model
    let config = BiRefNetConfig::swin_l();
    let model = BiRefNet::new(config, vb)?;

    // Run inference (input: 1024x1024 normalized image)
    let input = Tensor::randn(0., 1., (1, 3, 1024, 1024), &device)?;
    let logits = model.forward_logits(&input)?;
    let mask = candle_nn::ops::sigmoid(&logits)?;

    Ok(())
}
```

## Model Weights

Download BiRefNet weights from HuggingFace:
- [BiRefNet-general](https://huggingface.co/ZhengPeng7/BiRefNet)
- [BiRefNet-portrait](https://huggingface.co/ZhengPeng7/BiRefNet-portrait)

## Requirements

- macOS 14+ (Sonoma) or macOS 15+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Rust 1.70+

## Credits

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - Original PyTorch implementation
- [candle](https://github.com/huggingface/candle) - Rust ML framework
- [mps-flash-attention](https://github.com/mpsops/mps-flash-attention) - Flash attention for Metal
- [mps-deform-conv](https://github.com/mpsops/mps-deform-conv) - Deformable convolution for Metal
- [candle-mps-flash-attention](https://github.com/mpsops/candle-mps-flash-attention) - Rust bindings for MPS flash attention

## License

MIT

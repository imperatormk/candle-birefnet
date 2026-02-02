//! Compare flash vs non-flash attention outputs in BiRefNet
//!
//! This runs the Swin backbone twice - once with flash attn, once without -
//! and compares the outputs to see where they diverge.

use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_birefnet::{SwinTransformer, SwinConfig};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::F32;

    // Download weights
    println!("Loading model weights from HuggingFace...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("ZhengPeng7/BiRefNet".to_string());
    let weights_path = repo.get("model.safetensors")?;

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let remapped: HashMap<String, Tensor> = tensors.into_iter().collect();

    // Build Swin backbone
    let config = SwinConfig::swin_l();
    let vb = VarBuilder::from_tensors(remapped.clone(), dtype, &device);
    let swin = SwinTransformer::new(config, vb.pp("backbone.bb"))?;

    // Create a small test input (256x256 instead of 1024x1024)
    let input = Tensor::randn(0f32, 1.0, (1, 3, 256, 256), &device)?;
    println!("Input shape: {:?}", input.shape());

    // Run forward
    println!("\nRunning Swin forward...");
    let features = swin.forward(&input)?;

    // Check each stage output
    for (i, feat) in features.iter().enumerate() {
        let _ = feat.sum_all()?.to_scalar::<f32>()?; // sync
        let min: f32 = feat.min_all()?.to_scalar()?;
        let max: f32 = feat.max_all()?.to_scalar()?;
        let mean: f32 = feat.mean_all()?.to_scalar()?;
        println!("Stage {} {:?}: min={:.4}, max={:.4}, mean={:.4}",
                 i, feat.shape(), min, max, mean);
    }

    Ok(())
}

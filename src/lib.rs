//! BiRefNet - Bilateral Reference Network for image segmentation
//!
//! This is a candle port of BiRefNet with Metal deformable convolution support.
//! https://github.com/ZhengPeng7/BiRefNet

pub mod deform_conv;
pub mod decoder;
pub mod aspp;
pub mod birefnet;
pub mod swin;

pub use birefnet::BiRefNet;
pub use deform_conv::DeformableConv2d;
pub use swin::{SwinTransformer, SwinConfig};

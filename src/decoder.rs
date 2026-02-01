//! Decoder blocks for BiRefNet
//!
//! Contains BasicDecBlk, BasicLatBlk, ResBlk, and SimpleConvs modules.

use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, VarBuilder};

use crate::aspp::ASPPDeformable;

/// Configuration for decoder blocks
#[derive(Clone)]
pub struct DecoderConfig {
    pub use_aspp_deformable: bool,
    pub inter_channels_adaptive: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            use_aspp_deformable: true,
            inter_channels_adaptive: false, // BiRefNet uses fixed 64 inter channels
        }
    }
}

/// Simple convolution block (used for ipt_blk)
/// conv1 -> conv_out (NO activation between!)
pub struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        inter_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(in_channels, inter_channels, 3, cfg, vb.pp("conv1"))?;
        let conv_out = conv2d(inter_channels, out_channels, 3, cfg, vb.pp("conv_out"))?;
        Ok(Self { conv1, conv_out })
    }
}

impl Module for SimpleConvs {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Note: Python BiRefNet SimpleConvs has NO activation between conv1 and conv_out
        let x = self.conv1.forward(x)?;
        self.conv_out.forward(&x)
    }
}

/// Basic lateral block - 1x1 conv for channel projection
pub struct BasicLatBlk {
    conv: Conv2d,
}

impl BasicLatBlk {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = conv2d(in_channels, out_channels, 1, Default::default(), vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for BasicLatBlk {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Basic decoder block with BatchNorm and optional ASPP
/// conv_in -> bn_in -> relu -> [aspp] -> conv_out -> bn_out
pub struct BasicDecBlk {
    pub conv_in: Conv2d,
    pub bn_in: BatchNorm,
    pub dec_att: Option<ASPPDeformable>,
    pub conv_out: Conv2d,
    pub bn_out: BatchNorm,
}

impl BasicDecBlk {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        config: &DecoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inter_channels = if config.inter_channels_adaptive {
            in_channels / 4
        } else {
            64
        };

        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv_in = conv2d(in_channels, inter_channels, 3, cfg, vb.pp("conv_in"))?;
        let bn_in = batch_norm(inter_channels, 1e-5, vb.pp("bn_in"))?;

        let dec_att = if config.use_aspp_deformable {
            Some(ASPPDeformable::new(inter_channels, None, vb.pp("dec_att"))?)
        } else {
            None
        };

        let conv_out = conv2d(inter_channels, out_channels, 3, cfg, vb.pp("conv_out"))?;
        let bn_out = batch_norm(out_channels, 1e-5, vb.pp("bn_out"))?;

        Ok(Self {
            conv_in,
            bn_in,
            dec_att,
            conv_out,
            bn_out,
        })
    }
}

impl Module for BasicDecBlk {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_in.forward(x)?;
        let x = self.bn_in.forward_t(&x, false)?;
        let x = x.relu()?;

        let x = if let Some(ref dec_att) = self.dec_att {
            dec_att.forward(&x)?
        } else {
            x
        };

        let x = self.conv_out.forward(&x)?;
        self.bn_out.forward_t(&x, false)
    }
}

/// Residual block with skip connection
#[allow(dead_code)]
pub struct ResBlk {
    conv_in: Conv2d,
    bn_in: BatchNorm,
    dec_att: Option<ASPPDeformable>,
    conv_out: Conv2d,
    bn_out: BatchNorm,
    conv_resi: Conv2d,
}

impl ResBlk {
    #[allow(dead_code)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        config: &DecoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inter_channels = if config.inter_channels_adaptive {
            in_channels / 4
        } else {
            64
        };

        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv_in = conv2d(in_channels, inter_channels, 3, cfg, vb.pp("conv_in"))?;
        let bn_in = batch_norm(inter_channels, 1e-5, vb.pp("bn_in"))?;

        let dec_att = if config.use_aspp_deformable {
            Some(ASPPDeformable::new(inter_channels, None, vb.pp("dec_att"))?)
        } else {
            None
        };

        let conv_out = conv2d(inter_channels, out_channels, 3, cfg, vb.pp("conv_out"))?;
        let bn_out = batch_norm(out_channels, 1e-5, vb.pp("bn_out"))?;

        let conv_resi = conv2d(in_channels, out_channels, 1, Default::default(), vb.pp("conv_resi"))?;

        Ok(Self {
            conv_in,
            bn_in,
            dec_att,
            conv_out,
            bn_out,
            conv_resi,
        })
    }
}

impl Module for ResBlk {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let resi = self.conv_resi.forward(x)?;

        let x = self.conv_in.forward(x)?;
        let x = self.bn_in.forward_t(&x, false)?;
        let x = x.relu()?;

        let x = if let Some(ref dec_att) = self.dec_att {
            dec_att.forward(&x)?
        } else {
            x
        };

        let x = self.conv_out.forward(&x)?;
        let x = self.bn_out.forward_t(&x, false)?;

        x + resi
    }
}

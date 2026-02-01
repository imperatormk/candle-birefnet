//! BiRefNet - Bilateral Reference Network
//!
//! Full model architecture for image segmentation matching the pretrained weights.

use candle_core::{Result, Tensor, Module};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, seq, Sequential};

use crate::decoder::{BasicDecBlk, BasicLatBlk, DecoderConfig, SimpleConvs};
use crate::swin::{SwinTransformer, SwinConfig};

/// BiRefNet configuration
#[derive(Clone)]
pub struct BiRefNetConfig {
    /// Input image size (width, height)
    pub size: (usize, usize),
    /// Backbone type
    pub backbone: String,
    /// Base backbone channels [x1, x2, x3, x4] from shallowest to deepest
    pub backbone_channels: Vec<usize>,
    /// Use multi-scale input (doubles channels via concatenation)
    pub mul_scl_ipt: bool,
    /// Use multi-scale supervision
    pub ms_supervision: bool,
    /// Use decoder input patches
    pub dec_ipt: bool,
    /// Use deformable ASPP
    pub use_aspp_deformable: bool,
    /// Context features (concatenate x1,x2,x3 to x4)
    pub cxt: Vec<usize>,
}

impl Default for BiRefNetConfig {
    fn default() -> Self {
        Self {
            size: (1024, 1024),
            backbone: "swin_v1_l".to_string(),
            // swin_v1_l base channels: [192, 384, 768, 1536]
            backbone_channels: vec![192, 384, 768, 1536],
            mul_scl_ipt: true, // Doubles channels
            ms_supervision: true,
            dec_ipt: true,
            use_aspp_deformable: true,
            cxt: vec![192, 384, 768], // Concatenate x1,x2,x3 to x4
        }
    }
}

impl BiRefNetConfig {
    /// Get effective lateral channels (with mul_scl_ipt doubling)
    pub fn lateral_channels(&self) -> Vec<usize> {
        let mult = if self.mul_scl_ipt { 2 } else { 1 };
        self.backbone_channels.iter().map(|c| c * mult).collect()
    }

    /// Get x4 input channels (including cxt concatenation)
    pub fn x4_channels(&self) -> usize {
        let mult = if self.mul_scl_ipt { 2 } else { 1 };
        let base = self.backbone_channels[3] * mult;
        let cxt_sum: usize = self.cxt.iter().map(|c| c * mult).sum();
        base + cxt_sum
    }

    /// Configuration for swin_l backbone (default, matching pretrained weights)
    pub fn swin_l() -> Self {
        Self::default()
    }
}

/// Squeeze module - compresses combined features
pub struct SqueezeModule {
    pub blocks: Vec<BasicDecBlk>,
}

impl SqueezeModule {
    pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let config = DecoderConfig {
            use_aspp_deformable: true,
            inter_channels_adaptive: false,
        };
        // squeeze_module.0
        let block = BasicDecBlk::new(in_channels, out_channels, &config, vb.pp("0"))?;
        Ok(Self { blocks: vec![block] })
    }
}

impl Module for SqueezeModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for block in &self.blocks {
            out = block.forward(&out)?;
        }
        Ok(out)
    }
}

/// GDT (Gradient Detail) convolutions - conv + batchnorm
pub struct GdtConvs {
    conv: Conv2d,
    bn: candle_nn::BatchNorm,
}

impl GdtConvs {
    pub fn new(in_channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv = candle_nn::conv2d(in_channels, 16, 3, cfg, vb.pp("0"))?;
        let bn = candle_nn::batch_norm(16, 1e-5, vb.pp("1"))?;
        Ok(Self { conv, bn })
    }
}

impl Module for GdtConvs {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use candle_core::ModuleT;
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        x.relu()
    }
}

/// BiRefNet Decoder with full architecture
pub struct BiRefNetDecoder {
    #[allow(dead_code)]
    config: BiRefNetConfig,

    // Input blocks (ipt_blk1-5)
    pub ipt_blk1: SimpleConvs,
    pub ipt_blk2: SimpleConvs,
    pub ipt_blk3: SimpleConvs,
    pub ipt_blk4: SimpleConvs,
    pub ipt_blk5: SimpleConvs,

    // Decoder blocks
    pub decoder_block4: BasicDecBlk,
    pub decoder_block3: BasicDecBlk,
    pub decoder_block2: BasicDecBlk,
    pub decoder_block1: BasicDecBlk,

    // Lateral blocks
    pub lateral_block4: BasicLatBlk,
    pub lateral_block3: BasicLatBlk,
    pub lateral_block2: BasicLatBlk,

    // GDT convolutions (applied during inference for attention weighting)
    pub gdt_convs_4: GdtConvs,
    pub gdt_convs_3: GdtConvs,
    pub gdt_convs_2: GdtConvs,
    pub gdt_convs_attn_4: Sequential,
    pub gdt_convs_attn_3: Sequential,
    pub gdt_convs_attn_2: Sequential,
    #[allow(dead_code)]
    gdt_convs_pred_4: Sequential,
    #[allow(dead_code)]
    gdt_convs_pred_3: Sequential,
    #[allow(dead_code)]
    gdt_convs_pred_2: Sequential,

    // Output conv (Sequential with index 0)
    pub conv_out1: Sequential,

    // Multi-scale supervision heads (loaded for weight compatibility)
    #[allow(dead_code)]
    conv_ms_spvn_4: Conv2d,
    #[allow(dead_code)]
    conv_ms_spvn_3: Conv2d,
    #[allow(dead_code)]
    conv_ms_spvn_2: Conv2d,
}

impl BiRefNetDecoder {
    pub fn new(config: BiRefNetConfig, vb: VarBuilder) -> Result<Self> {
        let decoder_config = DecoderConfig {
            use_aspp_deformable: config.use_aspp_deformable,
            inter_channels_adaptive: false,
        };

        let lat_ch = config.lateral_channels();
        // lat_ch = [384, 768, 1536, 3072] with mul_scl_ipt

        // ipt_blk output channels
        let ipt_out = vec![48, 96, 192, 384, 384]; // ipt_blk1-5 outputs

        // ipt_blk input channels (based on weights analysis)
        // ipt_blk1: 3 -> 48
        // ipt_blk2: 48 -> 96
        // ipt_blk3: 192 -> 192 (lat_ch[0]/2 = 384/2)
        // ipt_blk4: 768 -> 384 (lat_ch[2]/2 = 1536/2)
        // ipt_blk5: 3072 -> 384 (lat_ch[3])

        let ipt_blk1 = SimpleConvs::new(3, ipt_out[0], 64, vb.pp("ipt_blk1"))?;
        let ipt_blk2 = SimpleConvs::new(ipt_out[0], ipt_out[1], 64, vb.pp("ipt_blk2"))?;
        let ipt_blk3 = SimpleConvs::new(lat_ch[0] / 2, ipt_out[2], 64, vb.pp("ipt_blk3"))?;
        let ipt_blk4 = SimpleConvs::new(lat_ch[2] / 2, ipt_out[3], 64, vb.pp("ipt_blk4"))?;
        let ipt_blk5 = SimpleConvs::new(lat_ch[3], ipt_out[4], 64, vb.pp("ipt_blk5"))?;

        // Decoder block input channels (features + ipt_blk output):
        // decoder_block4: x4(3072) + ipt_blk5(384) = 3456
        // decoder_block3: dec4_out(1536) + ipt_blk4(384) = 1920
        // decoder_block2: dec3_out(768) + ipt_blk3(192) = 960
        // decoder_block1: dec2_out(384) + ipt_blk2(96) = 480

        // Decoder block output channels
        let dec_out = vec![lat_ch[2], lat_ch[1], lat_ch[0], lat_ch[0] / 2]; // [1536, 768, 384, 192]

        let dec_in_4 = lat_ch[3] + ipt_out[4]; // 3072 + 384 = 3456
        let dec_in_3 = dec_out[0] + ipt_out[3]; // 1536 + 384 = 1920
        let dec_in_2 = dec_out[1] + ipt_out[2]; // 768 + 192 = 960
        let dec_in_1 = dec_out[2] + ipt_out[1]; // 384 + 96 = 480

        let decoder_block4 = BasicDecBlk::new(dec_in_4, dec_out[0], &decoder_config, vb.pp("decoder_block4"))?;
        let decoder_block3 = BasicDecBlk::new(dec_in_3, dec_out[1], &decoder_config, vb.pp("decoder_block3"))?;
        let decoder_block2 = BasicDecBlk::new(dec_in_2, dec_out[2], &decoder_config, vb.pp("decoder_block2"))?;
        let decoder_block1 = BasicDecBlk::new(dec_in_1, dec_out[3], &decoder_config, vb.pp("decoder_block1"))?;

        // Lateral blocks (same in/out channels)
        let lateral_block4 = BasicLatBlk::new(lat_ch[2], lat_ch[2], vb.pp("lateral_block4"))?;
        let lateral_block3 = BasicLatBlk::new(lat_ch[1], lat_ch[1], vb.pp("lateral_block3"))?;
        let lateral_block2 = BasicLatBlk::new(lat_ch[0], lat_ch[0], vb.pp("lateral_block2"))?;

        // GDT convolutions
        let gdt_convs_4 = GdtConvs::new(dec_out[0], vb.pp("gdt_convs_4"))?;
        let gdt_convs_3 = GdtConvs::new(dec_out[1], vb.pp("gdt_convs_3"))?;
        let gdt_convs_2 = GdtConvs::new(dec_out[2], vb.pp("gdt_convs_2"))?;

        // GDT attention (1x1 conv -> sigmoid)
        let gdt_convs_attn_4 = seq().add(candle_nn::conv2d(16, 1, 1, Default::default(), vb.pp("gdt_convs_attn_4.0"))?);
        let gdt_convs_attn_3 = seq().add(candle_nn::conv2d(16, 1, 1, Default::default(), vb.pp("gdt_convs_attn_3.0"))?);
        let gdt_convs_attn_2 = seq().add(candle_nn::conv2d(16, 1, 1, Default::default(), vb.pp("gdt_convs_attn_2.0"))?);

        // GDT prediction (1x1 conv)
        let gdt_convs_pred_4 = seq().add(candle_nn::conv2d(16, 1, 1, Default::default(), vb.pp("gdt_convs_pred_4.0"))?);
        let gdt_convs_pred_3 = seq().add(candle_nn::conv2d(16, 1, 1, Default::default(), vb.pp("gdt_convs_pred_3.0"))?);
        let gdt_convs_pred_2 = seq().add(candle_nn::conv2d(16, 1, 1, Default::default(), vb.pp("gdt_convs_pred_2.0"))?);

        // Output conv (final dec_out channels -> 1)
        // conv_out1.0 -> 1x1 conv from 240 -> 1
        // 240 = dec_out[3] + ipt_blk1 = 192 + 48
        let final_ch = dec_out[3] + ipt_out[0]; // 192 + 48 = 240
        let conv_out1 = seq().add(candle_nn::conv2d(final_ch, 1, 1, Default::default(), vb.pp("conv_out1.0"))?);

        // Multi-scale supervision heads
        let conv_ms_spvn_4 = candle_nn::conv2d(dec_out[0], 1, 1, Default::default(), vb.pp("conv_ms_spvn_4"))?;
        let conv_ms_spvn_3 = candle_nn::conv2d(dec_out[1], 1, 1, Default::default(), vb.pp("conv_ms_spvn_3"))?;
        let conv_ms_spvn_2 = candle_nn::conv2d(dec_out[2], 1, 1, Default::default(), vb.pp("conv_ms_spvn_2"))?;

        Ok(Self {
            config,
            ipt_blk1,
            ipt_blk2,
            ipt_blk3,
            ipt_blk4,
            ipt_blk5,
            decoder_block4,
            decoder_block3,
            decoder_block2,
            decoder_block1,
            lateral_block4,
            lateral_block3,
            lateral_block2,
            gdt_convs_4,
            gdt_convs_3,
            gdt_convs_2,
            gdt_convs_attn_4,
            gdt_convs_attn_3,
            gdt_convs_attn_2,
            gdt_convs_pred_4,
            gdt_convs_pred_3,
            gdt_convs_pred_2,
            conv_out1,
            conv_ms_spvn_4,
            conv_ms_spvn_3,
            conv_ms_spvn_2,
        })
    }

    /// Forward pass through decoder
    /// x: original input image
    /// x1-x4: backbone features at different scales
    pub fn forward(&self, x: &Tensor, x1: &Tensor, x2: &Tensor, x3: &Tensor, x4: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = x.dims4()?;
        let (_, _, _h4, _w4) = x4.dims4()?;
        let (_, _, h3, w3) = x3.dims4()?;
        let (_, _, h2, w2) = x2.dims4()?;
        let (_, _, h1, w1) = x1.dims4()?;

        // image2patches: split image into patches and flatten channels
        // For x4 scale: grid = 32x32, so we get 32*32=1024 patches * 3 channels = 3072
        // Input shape [B, 3, H, W] -> [B, 3*grid_h*grid_w, H/grid_h, W/grid_w]
        fn image2patches(x: &Tensor, target_h: usize, target_w: usize) -> Result<Tensor> {
            let (b, c, h, w) = x.dims4()?;
            let grid_h = h / target_h;
            let grid_w = w / target_w;

            // Reshape: [B, C, H, W] -> [B, C, grid_h, target_h, grid_w, target_w]
            let x = x.reshape((b, c, grid_h, target_h, grid_w, target_w))?;
            // Permute: [B, C, grid_h, grid_w, target_h, target_w]
            let x = x.permute((0, 1, 2, 4, 3, 5))?;
            // Reshape: [B, C*grid_h*grid_w, target_h, target_w]
            let out_c = c * grid_h * grid_w;
            x.reshape((b, out_c, target_h, target_w))
        }

        // Create image patches for each scale
        // x4 scale: image -> patches with 3072 channels
        let patches5 = image2patches(x, h / 32, w / 32)?; // 3 * 32 * 32 = 3072
        let ipt5 = self.ipt_blk5.forward(&patches5)?;

        // x3 scale: 768 channels
        let patches4 = image2patches(x, h / 16, w / 16)?; // 3 * 16 * 16 = 768
        let ipt4 = self.ipt_blk4.forward(&patches4)?;

        // x2 scale: 192 channels
        let patches3 = image2patches(x, h / 8, w / 8)?; // 3 * 8 * 8 = 192
        let ipt3 = self.ipt_blk3.forward(&patches3)?;

        // x1 scale: 48 channels
        let patches2 = image2patches(x, h / 4, w / 4)?; // 3 * 4 * 4 = 48
        let ipt2 = self.ipt_blk2.forward(&patches2)?;

        // Full resolution: 3 channels
        let ipt1 = self.ipt_blk1.forward(x)?;

        // Decoder stage 4: concat x4 + ipt5
        let d4_in = Tensor::cat(&[x4, &ipt5], 1)?;
        let p4 = self.decoder_block4.forward(&d4_in)?;

        // Apply GDT attention (out_ref=True in Python config)
        let p4_gdt = self.gdt_convs_4.forward(&p4)?;
        let gdt_attn_4 = candle_nn::ops::sigmoid(&self.gdt_convs_attn_4.forward(&p4_gdt)?)?;
        let p4 = p4.broadcast_mul(&gdt_attn_4)?;

        // Upsample and add lateral (Python uses bilinear with align_corners=True)
        let p4_up = p4.upsample_bilinear2d(h3, w3, true)?;
        let lat4 = self.lateral_block4.forward(x3)?;
        let p3_in = (p4_up + lat4)?;

        // Decoder stage 3: concat p3_in + ipt4
        let ipt4_up = ipt4.upsample_bilinear2d(h3, w3, true)?;
        let d3_in = Tensor::cat(&[&p3_in, &ipt4_up], 1)?;
        let p3 = self.decoder_block3.forward(&d3_in)?;

        // Apply GDT attention
        let p3_gdt = self.gdt_convs_3.forward(&p3)?;
        let gdt_attn_3 = candle_nn::ops::sigmoid(&self.gdt_convs_attn_3.forward(&p3_gdt)?)?;
        let p3 = p3.broadcast_mul(&gdt_attn_3)?;

        // Upsample and add lateral
        let p3_up = p3.upsample_bilinear2d(h2, w2, true)?;
        let lat3 = self.lateral_block3.forward(x2)?;
        let p2_in = (p3_up + lat3)?;

        // Decoder stage 2: concat p2_in + ipt3
        let ipt3_up = ipt3.upsample_bilinear2d(h2, w2, true)?;
        let d2_in = Tensor::cat(&[&p2_in, &ipt3_up], 1)?;
        let p2 = self.decoder_block2.forward(&d2_in)?;

        // Apply GDT attention
        let p2_gdt = self.gdt_convs_2.forward(&p2)?;
        let gdt_attn_2 = candle_nn::ops::sigmoid(&self.gdt_convs_attn_2.forward(&p2_gdt)?)?;
        let p2 = p2.broadcast_mul(&gdt_attn_2)?;

        // Upsample and add lateral
        let p2_up = p2.upsample_bilinear2d(h1, w1, true)?;
        let lat2 = self.lateral_block2.forward(x1)?;
        let p1_in = (p2_up + lat2)?;

        // Decoder stage 1: concat p1_in + ipt2
        let ipt2_up = ipt2.upsample_bilinear2d(h1, w1, true)?;
        let d1_in = Tensor::cat(&[&p1_in, &ipt2_up], 1)?;
        let p1 = self.decoder_block1.forward(&d1_in)?;

        // Final output: upsample p1 to original size, concat with ipt1
        let p1_up = p1.upsample_bilinear2d(h, w, true)?;
        let ipt1_up = ipt1.upsample_bilinear2d(h, w, true)?;
        let final_in = Tensor::cat(&[&p1_up, &ipt1_up], 1)?;
        self.conv_out1.forward(&final_in)
    }
}

/// BiRefNet model with Swin Transformer backbone
pub struct BiRefNet {
    pub config: BiRefNetConfig,
    pub backbone: SwinTransformer,
    pub squeeze_module: SqueezeModule,
    pub decoder: BiRefNetDecoder,
}

impl BiRefNet {
    /// Create a new BiRefNet model matching pretrained weights
    pub fn new(config: BiRefNetConfig, vb: VarBuilder) -> Result<Self> {
        // Always use swin_l for now (matching pretrained)
        let swin_config = SwinConfig::swin_l();

        let backbone = SwinTransformer::new(swin_config, vb.pp("bb"))?;

        // squeeze_module input: x4_channels (with cxt concatenation)
        // = 3072 (doubled x4) + 384 + 768 + 1536 (doubled cxt) = 5760
        let squeeze_in = config.x4_channels();
        let squeeze_out = config.lateral_channels()[3]; // 3072
        let squeeze_module = SqueezeModule::new(squeeze_in, squeeze_out, vb.pp("squeeze_module"))?;

        let decoder = BiRefNetDecoder::new(config.clone(), vb.pp("decoder"))?;

        Ok(Self {
            config,
            backbone,
            squeeze_module,
            decoder,
        })
    }

    /// Forward pass returning raw logits (no sigmoid)
    pub fn forward_logits(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;

        // Get backbone features
        let features = self.backbone.forward(x)?;
        let mut x1 = features[0].clone(); // H/4, C=192
        let mut x2 = features[1].clone(); // H/8, C=384
        let mut x3 = features[2].clone(); // H/16, C=768
        let mut x4 = features[3].clone(); // H/32, C=1536

        // mul_scl_ipt: run backbone on half-scale input and concatenate
        if self.config.mul_scl_ipt {
            // Python uses bilinear with align_corners=True for downsampling
            let x_half = x.upsample_bilinear2d(h / 2, w / 2, true)?;
            let features_half = self.backbone.forward(&x_half)?;

            // Upsample half-scale features to match full-scale and concatenate
            let (_, _, h1, w1) = x1.dims4()?;
            let (_, _, h2, w2) = x2.dims4()?;
            let (_, _, h3, w3) = x3.dims4()?;
            let (_, _, h4, w4) = x4.dims4()?;

            // Python uses bilinear with align_corners=True
            let x1_half = features_half[0].upsample_bilinear2d(h1, w1, true)?;
            let x2_half = features_half[1].upsample_bilinear2d(h2, w2, true)?;
            let x3_half = features_half[2].upsample_bilinear2d(h3, w3, true)?;
            let x4_half = features_half[3].upsample_bilinear2d(h4, w4, true)?;

            x1 = Tensor::cat(&[&x1, &x1_half], 1)?;
            x2 = Tensor::cat(&[&x2, &x2_half], 1)?;
            x3 = Tensor::cat(&[&x3, &x3_half], 1)?;
            x4 = Tensor::cat(&[&x4, &x4_half], 1)?;
        }

        // cxt: concatenate upsampled x1,x2,x3 to x4
        if !self.config.cxt.is_empty() {
            let (_, _, h4, w4) = x4.dims4()?;
            // Python uses bilinear with align_corners=True
            let x1_to_x4 = x1.upsample_bilinear2d(h4, w4, true)?;
            let x2_to_x4 = x2.upsample_bilinear2d(h4, w4, true)?;
            let x3_to_x4 = x3.upsample_bilinear2d(h4, w4, true)?;
            x4 = Tensor::cat(&[&x1_to_x4, &x2_to_x4, &x3_to_x4, &x4], 1)?;
        }

        // squeeze_module compresses x4
        let x4 = self.squeeze_module.forward(&x4)?;

        // Decoder
        self.decoder.forward(x, &x1, &x2, &x3, &x4)
    }
}

impl BiRefNet {
    /// Full forward pass with sigmoid activation (returns [0, 1] mask)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(x)?;
        candle_nn::ops::sigmoid(&logits)
    }
}

impl Module for BiRefNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

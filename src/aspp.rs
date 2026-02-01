//! ASPP (Atrous Spatial Pyramid Pooling) modules for BiRefNet
//!
//! Matches the pretrained weight structure exactly.

use candle_core::{Result, Tensor, Module, ModuleT, D, DType};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, BatchNorm, batch_norm};

#[cfg(feature = "metal")]
use candle_core::Device;

/// Deformable convolution matching pretrained weights structure
/// Has: offset_conv, modulator_conv, regular_conv (no bias)
pub struct DeformConvASPP {
    pub offset_conv: Conv2d,
    pub modulator_conv: Conv2d,
    pub regular_conv: Conv2d,
    kernel_size: usize,
    padding: usize,
    in_channels: usize,
    out_channels: usize,
}

impl DeformConvASPP {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kk = kernel_size * kernel_size;

        let cfg = Conv2dConfig {
            padding,
            ..Default::default()
        };

        // offset_conv: predicts 2 * k * k offsets
        let offset_conv = candle_nn::conv2d(in_channels, 2 * kk, kernel_size, cfg, vb.pp("offset_conv"))?;

        // modulator_conv: predicts k * k modulators
        let modulator_conv = candle_nn::conv2d(in_channels, kk, kernel_size, cfg, vb.pp("modulator_conv"))?;

        // regular_conv: NO BIAS (use conv2d_no_bias)
        let regular_conv = candle_nn::conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb.pp("regular_conv"))?;

        Ok(Self {
            offset_conv,
            modulator_conv,
            regular_conv,
            kernel_size,
            padding,
            in_channels,
            out_channels,
        })
    }

    #[cfg(feature = "metal")]
    fn forward_metal(&self, x: &Tensor, offset: &Tensor, mask: &Tensor) -> Result<Tensor> {
        use candle_metal_kernels::{DeformConv2dConfig, call_deformable_im2col, BufferOffset};

        let device = x.device();
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return self.regular_conv.forward(x),
        };

        let (batch, channels, height, width) = x.dims4()?;
        let weight = self.regular_conv.weight();
        let (out_channels, _, kh, kw) = weight.dims4()?;

        // Calculate output dimensions
        let out_h = (height + 2 * self.padding - kh) / 1 + 1; // stride=1
        let out_w = (width + 2 * self.padding - kw) / 1 + 1;

        // Create config
        let cfg = DeformConv2dConfig::new(
            height,
            width,
            kh,
            kw,
            (self.padding, self.padding),
            (1, 1), // stride
            (1, 1), // dilation
            batch,
            channels,
            1, // n_offset_grps
            true, // use_mask
        );

        // Create output columns buffer
        // columns shape: [C * kH * kW, batch * out_H * out_W]
        let col_size = channels * kh * kw * batch * out_h * out_w;
        let columns = Tensor::zeros((col_size,), x.dtype(), device)?;

        // Get the kernel name based on dtype
        let kernel_name = match x.dtype() {
            DType::F32 => "deformable_im2col_f32",
            DType::F16 => "deformable_im2col_f16",
            DType::BF16 => "deformable_im2col_bf16",
            dt => candle_core::bail!("Unsupported dtype for deform_conv: {:?}", dt),
        };

        // Get metal device components
        let mtl_device = metal_device.device();
        let kernels = metal_device.kernels();

        // Get buffer info in a scope so guards are dropped before encoder call
        let (x_buf, x_off, offset_buf, offset_off, mask_buf, mask_off, col_buf) = {
            let (x_storage, x_layout) = x.storage_and_layout();
            let (offset_storage, offset_layout) = offset.storage_and_layout();
            let (mask_storage, mask_layout) = mask.storage_and_layout();
            let (col_storage, _col_layout) = columns.storage_and_layout();

            let (x_buf, x_off) = match &*x_storage {
                candle_core::Storage::Metal(s) => (s.buffer().clone(), x_layout.start_offset() * x.dtype().size_in_bytes()),
                _ => candle_core::bail!("Expected Metal storage for input"),
            };
            let (offset_buf, offset_off) = match &*offset_storage {
                candle_core::Storage::Metal(s) => (s.buffer().clone(), offset_layout.start_offset() * offset.dtype().size_in_bytes()),
                _ => candle_core::bail!("Expected Metal storage for offset"),
            };
            let (mask_buf, mask_off) = match &*mask_storage {
                candle_core::Storage::Metal(s) => (s.buffer().clone(), mask_layout.start_offset() * mask.dtype().size_in_bytes()),
                _ => candle_core::bail!("Expected Metal storage for mask"),
            };
            let col_buf = match &*col_storage {
                candle_core::Storage::Metal(s) => s.buffer().clone(),
                _ => candle_core::bail!("Expected Metal storage for columns"),
            };
            (x_buf, x_off, offset_buf, offset_off, mask_buf, mask_off, col_buf)
        };

        // Now get encoder after releasing storage guards
        let encoder = metal_device.command_encoder()?;

        // Call the kernel
        call_deformable_im2col(
            mtl_device,
            &encoder,
            kernels,
            kernel_name,
            &cfg,
            BufferOffset { buffer: &x_buf, offset_in_bytes: x_off },
            BufferOffset { buffer: &offset_buf, offset_in_bytes: offset_off },
            BufferOffset { buffer: &mask_buf, offset_in_bytes: mask_off },
            &col_buf,
        ).map_err(|e| candle_core::Error::Msg(format!("Metal kernel error: {:?}", e)))?;

        // Drop encoder to ensure kernel is scheduled
        drop(encoder);

        // Reshape columns for matmul: [C*kH*kW, B*outH*outW]
        let columns = columns.reshape((channels * kh * kw, batch * out_h * out_w))?;

        // Reshape weight: [outC, C, kH, kW] -> [outC, C*kH*kW]
        let weight = weight.reshape((out_channels, channels * kh * kw))?;

        // matmul: [outC, C*kH*kW] @ [C*kH*kW, B*outH*outW] -> [outC, B*outH*outW]
        let out = weight.matmul(&columns)?;

        // Reshape to [B, outC, outH, outW]
        let out = out.reshape((out_channels, batch, out_h, out_w))?;
        out.permute((1, 0, 2, 3))
    }
}

impl Module for DeformConvASPP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute offsets and modulators
        let offset = self.offset_conv.forward(x)?;
        // Sigmoid * 2 for modulator: 2 / (1 + exp(-x))
        let mod_raw = self.modulator_conv.forward(x)?;
        let mask = ((mod_raw.neg()?.exp()? + 1.0)?.recip()? * 2.0)?;

        #[cfg(feature = "metal")]
        {
            if let Device::Metal(_) = x.device() {
                return self.forward_metal(x, &offset, &mask);
            }
        }

        // CPU fallback: just use regular conv (ignores offsets/mask)
        let _ = (&offset, &mask);
        self.regular_conv.forward(x)
    }
}

/// ASPP module with deformable conv and BatchNorm
pub struct ASPPModuleDeformable {
    pub atrous_conv: DeformConvASPP,
    pub bn: BatchNorm,
}

impl ASPPModuleDeformable {
    pub fn new(
        in_channels: usize,
        planes: usize,
        kernel_size: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let atrous_conv = DeformConvASPP::new(
            in_channels,
            planes,
            kernel_size,
            padding,
            vb.pp("atrous_conv"),
        )?;

        let bn = batch_norm(planes, 1e-5, vb.pp("bn"))?;

        Ok(Self { atrous_conv, bn })
    }
}

impl Module for ASPPModuleDeformable {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.atrous_conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        x.relu()
    }
}

/// Deformable ASPP - matches pretrained structure exactly
/// Has: aspp1, aspp_deforms[0-2], global_avg_pool, conv1, bn1
pub struct ASPPDeformable {
    pub aspp1: ASPPModuleDeformable,
    pub aspp_deforms: Vec<ASPPModuleDeformable>,
    pub global_avg_pool_conv: Conv2d,
    pub global_avg_pool_bn: BatchNorm,
    pub conv1: Conv2d,
    pub bn1: BatchNorm,
}

impl ASPPDeformable {
    pub fn new(
        in_channels: usize,
        out_channels: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(in_channels);
        let inter_channels = 256;
        let parallel_block_sizes = [1, 3, 7];

        // aspp1: 1x1 conv
        let aspp1 = ASPPModuleDeformable::new(
            in_channels,
            inter_channels,
            1,
            0,
            vb.pp("aspp1"),
        )?;

        // aspp_deforms: 1x1, 3x3, 7x7
        let mut aspp_deforms = Vec::new();
        for (i, &conv_size) in parallel_block_sizes.iter().enumerate() {
            let padding = conv_size / 2;
            let module = ASPPModuleDeformable::new(
                in_channels,
                inter_channels,
                conv_size,
                padding,
                vb.pp("aspp_deforms").pp(i),
            )?;
            aspp_deforms.push(module);
        }

        // global_avg_pool: Sequential(AdaptiveAvgPool2d, Conv2d_no_bias, BatchNorm, ReLU)
        // global_avg_pool.1 = conv (no bias), global_avg_pool.2 = bn
        let global_avg_pool_conv = candle_nn::conv2d_no_bias(
            in_channels,
            inter_channels,
            1,
            Default::default(),
            vb.pp("global_avg_pool").pp("1"),
        )?;
        let global_avg_pool_bn = batch_norm(inter_channels, 1e-5, vb.pp("global_avg_pool").pp("2"))?;

        // conv1: combines all branches
        // Total branches: aspp1 + 3 aspp_deforms + global_avg_pool = 5
        let conv1 = candle_nn::conv2d_no_bias(
            inter_channels * 5,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv1"),
        )?;

        let bn1 = batch_norm(out_channels, 1e-5, vb.pp("bn1"))?;

        Ok(Self {
            aspp1,
            aspp_deforms,
            global_avg_pool_conv,
            global_avg_pool_bn,
            conv1,
            bn1,
        })
    }
}

impl Module for ASPPDeformable {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.aspp1.forward(x)?;

        let mut deform_outputs = Vec::new();
        for aspp in &self.aspp_deforms {
            deform_outputs.push(aspp.forward(x)?);
        }

        // Global average pooling
        let (_, _, h, w) = x.dims4()?;
        let x5 = x.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let x5 = self.global_avg_pool_conv.forward(&x5)?;
        let x5 = self.global_avg_pool_bn.forward_t(&x5, false)?;
        let x5 = x5.relu()?;
        let x5 = x5.upsample_nearest2d(h, w)?;

        // Concatenate all: aspp1 + 3 deforms + global_avg_pool = 5 branches
        let mut to_cat: Vec<&Tensor> = vec![&x1];
        for t in &deform_outputs {
            to_cat.push(t);
        }
        to_cat.push(&x5);

        let out = Tensor::cat(&to_cat, 1)?;

        let out = self.conv1.forward(&out)?;
        let out = self.bn1.forward_t(&out, false)?;
        out.relu()
    }
}

/// Regular ASPP module (non-deformable) - keeping for reference
#[allow(dead_code)]
pub struct ASPPModule {
    atrous_conv: Conv2d,
}

#[allow(dead_code)]
impl ASPPModule {
    pub fn new(
        in_channels: usize,
        planes: usize,
        kernel_size: usize,
        padding: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding,
            dilation,
            ..Default::default()
        };

        let atrous_conv = candle_nn::conv2d(
            in_channels,
            planes,
            kernel_size,
            conv_cfg,
            vb.pp("atrous_conv"),
        )?;

        Ok(Self { atrous_conv })
    }
}

impl Module for ASPPModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.atrous_conv.forward(x)?.relu()
    }
}

/// Regular ASPP with multiple dilation rates - keeping for reference
#[allow(dead_code)]
pub struct ASPP {
    aspp1: ASPPModule,
    aspp2: ASPPModule,
    aspp3: ASPPModule,
    aspp4: ASPPModule,
    global_avg_pool_conv: Conv2d,
    conv1: Conv2d,
}

#[allow(dead_code)]
impl ASPP {
    pub fn new(
        in_channels: usize,
        out_channels: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(in_channels);
        let inter_channels = 256;
        let dilations = [1, 6, 12, 18];

        let aspp1 = ASPPModule::new(in_channels, inter_channels, 1, 0, dilations[0], vb.pp("aspp1"))?;
        let aspp2 = ASPPModule::new(in_channels, inter_channels, 3, dilations[1], dilations[1], vb.pp("aspp2"))?;
        let aspp3 = ASPPModule::new(in_channels, inter_channels, 3, dilations[2], dilations[2], vb.pp("aspp3"))?;
        let aspp4 = ASPPModule::new(in_channels, inter_channels, 3, dilations[3], dilations[3], vb.pp("aspp4"))?;

        let global_avg_pool_conv = candle_nn::conv2d(
            in_channels,
            inter_channels,
            1,
            Default::default(),
            vb.pp("global_avg_pool").pp("1"),
        )?;

        let conv1 = candle_nn::conv2d(
            inter_channels * 5,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv1"),
        )?;

        Ok(Self {
            aspp1,
            aspp2,
            aspp3,
            aspp4,
            global_avg_pool_conv,
            conv1,
        })
    }
}

impl Module for ASPP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.aspp1.forward(x)?;
        let x2 = self.aspp2.forward(x)?;
        let x3 = self.aspp3.forward(x)?;
        let x4 = self.aspp4.forward(x)?;

        // Global average pooling
        let (_, _, h, w) = x.dims4()?;
        let x5 = x.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let x5 = self.global_avg_pool_conv.forward(&x5)?.relu()?;
        let x5 = x5.upsample_nearest2d(h, w)?;

        // Concatenate
        let out = Tensor::cat(&[&x1, &x2, &x3, &x4, &x5], 1)?;

        self.conv1.forward(&out)?.relu()
    }
}

//! Deformable Convolution 2D for candle
//!
//! This module provides a DeformableConv2d layer that uses learned offsets
//! to sample input features at irregular positions.

use candle_core::{Result, Tensor, Module, DType};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

#[cfg(feature = "metal")]
use candle_core::Device;

/// Deformable Convolution 2D layer
///
/// Similar to regular Conv2d but with learnable offsets that allow
/// the convolution to sample at irregular positions.
#[allow(dead_code)]
pub struct DeformableConv2d {
    offset_conv: Conv2d,
    modulator_conv: Conv2d,
    regular_conv: Conv2d,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    in_channels: usize,
    out_channels: usize,
}

impl DeformableConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let offset_channels = 2 * kernel_size * kernel_size;
        let mask_channels = kernel_size * kernel_size;

        let conv_cfg = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };

        let offset_conv = candle_nn::conv2d(
            in_channels,
            offset_channels,
            kernel_size,
            conv_cfg,
            vb.pp("offset_conv"),
        )?;

        let modulator_conv = candle_nn::conv2d(
            in_channels,
            mask_channels,
            kernel_size,
            conv_cfg,
            vb.pp("modulator_conv"),
        )?;

        let regular_conv = candle_nn::conv2d(
            in_channels,
            out_channels,
            kernel_size,
            conv_cfg,
            vb.pp("regular_conv"),
        )?;

        Ok(Self {
            offset_conv,
            modulator_conv,
            regular_conv,
            kernel_size,
            padding,
            stride,
            in_channels,
            out_channels,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let offset = self.offset_conv.forward(x)?;
        // Sigmoid: 1 / (1 + exp(-x))
        let mod_raw = self.modulator_conv.forward(x)?;
        let modulator = ((mod_raw.neg()?.exp()? + 1.0)?.recip()? * 2.0)?;

        #[cfg(feature = "metal")]
        {
            if let Device::Metal(_) = x.device() {
                return self.forward_metal(x, &offset, &modulator);
            }
        }

        // Fallback: regular convolution (no deformation)
        // This loses the deformable property but allows the model to run
        let _ = (&offset, &modulator); // silence unused warnings
        self.regular_conv.forward(x)
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
        let out_h = (height + 2 * self.padding - kh) / self.stride + 1;
        let out_w = (width + 2 * self.padding - kw) / self.stride + 1;

        // Create config
        let cfg = DeformConv2dConfig::new(
            height,
            width,
            kh,
            kw,
            (self.padding, self.padding),
            (self.stride, self.stride),
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
        let encoder = metal_device.command_encoder()?;
        let kernels = metal_device.kernels();

        // Get storage references - we need to hold them while we use the buffers
        let (x_storage, x_layout) = x.storage_and_layout();
        let (offset_storage, offset_layout) = offset.storage_and_layout();
        let (mask_storage, mask_layout) = mask.storage_and_layout();
        let (col_storage, _col_layout) = columns.storage_and_layout();

        // Extract metal buffers from storage
        let (x_buf, x_off) = match &*x_storage {
            candle_core::Storage::Metal(s) => (s.buffer(), x_layout.start_offset() * x.dtype().size_in_bytes()),
            _ => candle_core::bail!("Expected Metal storage for input"),
        };
        let (offset_buf, offset_off) = match &*offset_storage {
            candle_core::Storage::Metal(s) => (s.buffer(), offset_layout.start_offset() * offset.dtype().size_in_bytes()),
            _ => candle_core::bail!("Expected Metal storage for offset"),
        };
        let (mask_buf, mask_off) = match &*mask_storage {
            candle_core::Storage::Metal(s) => (s.buffer(), mask_layout.start_offset() * mask.dtype().size_in_bytes()),
            _ => candle_core::bail!("Expected Metal storage for mask"),
        };
        let col_buf = match &*col_storage {
            candle_core::Storage::Metal(s) => s.buffer(),
            _ => candle_core::bail!("Expected Metal storage for columns"),
        };

        // Call the kernel
        call_deformable_im2col(
            mtl_device,
            &encoder,
            kernels,
            kernel_name,
            &cfg,
            BufferOffset { buffer: x_buf, offset_in_bytes: x_off },
            BufferOffset { buffer: offset_buf, offset_in_bytes: offset_off },
            BufferOffset { buffer: mask_buf, offset_in_bytes: mask_off },
            col_buf,
        ).map_err(|e| candle_core::Error::Msg(format!("Metal kernel error: {:?}", e)))?;

        // Drop storage refs before reshaping
        drop(x_storage);
        drop(offset_storage);
        drop(mask_storage);
        drop(col_storage);

        // Reshape columns for matmul: [C*kH*kW, B*outH*outW]
        let columns = columns.reshape((channels * kh * kw, batch * out_h * out_w))?;

        // Reshape weight: [outC, C, kH, kW] -> [outC, C*kH*kW]
        let weight = weight.reshape((out_channels, channels * kh * kw))?;

        // matmul: [outC, C*kH*kW] @ [C*kH*kW, B*outH*outW] -> [outC, B*outH*outW]
        let out = weight.matmul(&columns)?;

        // Reshape to [B, outC, outH, outW]
        let out = out.reshape((out_channels, batch, out_h, out_w))?;
        let out = out.permute((1, 0, 2, 3))?;

        // Add bias if present
        if let Some(bias) = self.regular_conv.bias() {
            let bias = bias.reshape((1, out_channels, 1, 1))?;
            return out.broadcast_add(&bias);
        }

        Ok(out)
    }
}

impl Module for DeformableConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

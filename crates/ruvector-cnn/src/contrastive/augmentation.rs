//! # Contrastive Augmentation
//!
//! SimCLR-style data augmentation for contrastive learning.
//!
//! ## Augmentation Pipeline
//!
//! The default SimCLR augmentation pipeline includes:
//! 1. Random resized crop (scale 0.08-1.0, ratio 3/4-4/3)
//! 2. Random horizontal flip (p=0.5)
//! 3. Color jitter (brightness, contrast, saturation, hue)
//! 4. Random grayscale (p=0.2)
//! 5. Gaussian blur (optional)
//!
//! ## References
//!
//! - SimCLR: "A Simple Framework for Contrastive Learning of Visual Representations"
//! - MoCo: "Momentum Contrast for Unsupervised Visual Representation Learning"

#[cfg(feature = "augmentation")]
use crate::error::{CnnError, CnnResult};
#[cfg(feature = "augmentation")]
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for contrastive augmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Minimum crop scale (default: 0.08)
    pub crop_scale_min: f64,
    /// Maximum crop scale (default: 1.0)
    pub crop_scale_max: f64,
    /// Minimum aspect ratio (default: 0.75)
    pub aspect_ratio_min: f64,
    /// Maximum aspect ratio (default: 1.333)
    pub aspect_ratio_max: f64,
    /// Probability of horizontal flip (default: 0.5)
    pub horizontal_flip_prob: f64,
    /// Brightness jitter factor (default: 0.4)
    pub brightness: f64,
    /// Contrast jitter factor (default: 0.4)
    pub contrast: f64,
    /// Saturation jitter factor (default: 0.4)
    pub saturation: f64,
    /// Hue jitter factor (default: 0.1)
    pub hue: f64,
    /// Probability of color jitter (default: 0.8)
    pub color_jitter_prob: f64,
    /// Probability of grayscale conversion (default: 0.2)
    pub grayscale_prob: f64,
    /// Gaussian blur kernel size (0 to disable)
    pub blur_kernel_size: u32,
    /// Probability of Gaussian blur (default: 0.5)
    pub blur_prob: f64,
    /// Gaussian blur sigma range
    pub blur_sigma_range: (f64, f64),
    /// Target output size (width, height)
    pub output_size: (u32, u32),
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            crop_scale_min: 0.08,
            crop_scale_max: 1.0,
            aspect_ratio_min: 0.75,
            aspect_ratio_max: 4.0 / 3.0,
            horizontal_flip_prob: 0.5,
            brightness: 0.4,
            contrast: 0.4,
            saturation: 0.4,
            hue: 0.1,
            color_jitter_prob: 0.8,
            grayscale_prob: 0.2,
            blur_kernel_size: 0,
            blur_prob: 0.5,
            blur_sigma_range: (0.1, 2.0),
            output_size: (224, 224),
        }
    }
}

/// Builder for ContrastiveAugmentation.
#[derive(Debug, Clone)]
pub struct ContrastiveAugmentationBuilder {
    config: AugmentationConfig,
    seed: Option<u64>,
}

impl ContrastiveAugmentationBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: AugmentationConfig::default(),
            seed: None,
        }
    }

    /// Set the crop scale range.
    pub fn crop_scale(mut self, min: f64, max: f64) -> Self {
        self.config.crop_scale_min = min;
        self.config.crop_scale_max = max;
        self
    }

    /// Set the aspect ratio range.
    pub fn aspect_ratio(mut self, min: f64, max: f64) -> Self {
        self.config.aspect_ratio_min = min;
        self.config.aspect_ratio_max = max;
        self
    }

    /// Set the horizontal flip probability.
    pub fn horizontal_flip_prob(mut self, prob: f64) -> Self {
        self.config.horizontal_flip_prob = prob;
        self
    }

    /// Set the color jitter parameters.
    pub fn color_jitter(mut self, brightness: f64, contrast: f64, saturation: f64, hue: f64) -> Self {
        self.config.brightness = brightness;
        self.config.contrast = contrast;
        self.config.saturation = saturation;
        self.config.hue = hue;
        self
    }

    /// Set the color jitter probability.
    pub fn color_jitter_prob(mut self, prob: f64) -> Self {
        self.config.color_jitter_prob = prob;
        self
    }

    /// Set the grayscale probability.
    pub fn grayscale_prob(mut self, prob: f64) -> Self {
        self.config.grayscale_prob = prob;
        self
    }

    /// Enable Gaussian blur with the specified kernel size.
    pub fn gaussian_blur(mut self, kernel_size: u32, sigma_range: (f64, f64)) -> Self {
        self.config.blur_kernel_size = kernel_size;
        self.config.blur_sigma_range = sigma_range;
        self
    }

    /// Set the blur probability.
    pub fn blur_prob(mut self, prob: f64) -> Self {
        self.config.blur_prob = prob;
        self
    }

    /// Set the output size.
    pub fn output_size(mut self, width: u32, height: u32) -> Self {
        self.config.output_size = (width, height);
        self
    }

    /// Set a fixed random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the ContrastiveAugmentation instance.
    pub fn build(self) -> ContrastiveAugmentation {
        let rng = if let Some(seed) = self.seed {
            rand::SeedableRng::seed_from_u64(seed)
        } else {
            rand::SeedableRng::from_entropy()
        };
        ContrastiveAugmentation {
            config: self.config,
            rng,
        }
    }
}

impl Default for ContrastiveAugmentationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// SimCLR-style contrastive augmentation pipeline.
///
/// # Example
///
/// ```rust,no_run
/// use ruvector_cnn::contrastive::ContrastiveAugmentation;
///
/// let aug = ContrastiveAugmentation::builder()
///     .crop_scale(0.08, 1.0)
///     .horizontal_flip_prob(0.5)
///     .color_jitter(0.4, 0.4, 0.4, 0.1)
///     .output_size(224, 224)
///     .build();
///
/// // Generate two augmented views of an image (requires augmentation feature)
/// // let (view1, view2) = aug.generate_pair(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct ContrastiveAugmentation {
    config: AugmentationConfig,
    rng: rand::rngs::StdRng,
}

impl ContrastiveAugmentation {
    /// Create a builder for ContrastiveAugmentation.
    pub fn builder() -> ContrastiveAugmentationBuilder {
        ContrastiveAugmentationBuilder::new()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AugmentationConfig {
        &self.config
    }

    /// Generate two augmented views of an image.
    ///
    /// This is the core operation for SimCLR-style contrastive learning.
    ///
    /// # Arguments
    ///
    /// * `image` - The input image
    ///
    /// # Returns
    ///
    /// A tuple of two independently augmented views.
    #[cfg(feature = "augmentation")]
    pub fn generate_pair(&mut self, image: &DynamicImage) -> CnnResult<(RgbImage, RgbImage)> {
        let view1 = self.augment(image)?;
        let view2 = self.augment(image)?;
        Ok((view1, view2))
    }

    /// Apply the full augmentation pipeline to an image.
    #[cfg(feature = "augmentation")]
    pub fn augment(&mut self, image: &DynamicImage) -> CnnResult<RgbImage> {
        let mut img = image.to_rgb8();

        // 1. Random resized crop
        img = self.random_resized_crop(&img)?;

        // 2. Random horizontal flip
        if self.rng.gen::<f64>() < self.config.horizontal_flip_prob {
            img = self.horizontal_flip(&img);
        }

        // 3. Color jitter (with probability)
        if self.rng.gen::<f64>() < self.config.color_jitter_prob {
            img = self.color_jitter(&img)?;
        }

        // 4. Random grayscale
        if self.rng.gen::<f64>() < self.config.grayscale_prob {
            img = self.to_grayscale(&img);
        }

        // 5. Gaussian blur (optional)
        if self.config.blur_kernel_size > 0 && self.rng.gen::<f64>() < self.config.blur_prob {
            img = self.gaussian_blur(&img)?;
        }

        Ok(img)
    }

    /// Random resized crop with configurable scale and aspect ratio.
    #[cfg(feature = "augmentation")]
    pub fn random_resized_crop(&mut self, image: &RgbImage) -> CnnResult<RgbImage> {
        let (orig_w, orig_h) = image.dimensions();
        let orig_area = (orig_w * orig_h) as f64;

        // Try up to 10 times to find a valid crop
        for _ in 0..10 {
            // Sample scale and aspect ratio
            let scale = self.rng.gen_range(self.config.crop_scale_min..=self.config.crop_scale_max);
            let aspect = self.rng.gen_range(
                self.config.aspect_ratio_min.ln()..=self.config.aspect_ratio_max.ln(),
            ).exp();

            // Compute crop dimensions
            let crop_area = orig_area * scale;
            let crop_w = (crop_area * aspect).sqrt() as u32;
            let crop_h = (crop_area / aspect).sqrt() as u32;

            if crop_w <= orig_w && crop_h <= orig_h && crop_w > 0 && crop_h > 0 {
                // Random position
                let x = self.rng.gen_range(0..=(orig_w - crop_w));
                let y = self.rng.gen_range(0..=(orig_h - crop_h));

                // Crop
                let cropped = image::imageops::crop_imm(image, x, y, crop_w, crop_h).to_image();

                // Resize to output size
                let (target_w, target_h) = self.config.output_size;
                let resized = image::imageops::resize(
                    &cropped,
                    target_w,
                    target_h,
                    image::imageops::FilterType::Lanczos3,
                );

                return Ok(resized);
            }
        }

        // Fallback: center crop to maintain aspect ratio, then resize
        let (target_w, target_h) = self.config.output_size;
        let target_ratio = target_w as f64 / target_h as f64;
        let orig_ratio = orig_w as f64 / orig_h as f64;

        let (crop_w, crop_h) = if orig_ratio > target_ratio {
            // Original is wider - crop width
            let h = orig_h;
            let w = (h as f64 * target_ratio) as u32;
            (w, h)
        } else {
            // Original is taller - crop height
            let w = orig_w;
            let h = (w as f64 / target_ratio) as u32;
            (w, h)
        };

        let x = (orig_w - crop_w) / 2;
        let y = (orig_h - crop_h) / 2;

        let cropped = image::imageops::crop_imm(image, x, y, crop_w, crop_h).to_image();
        let resized = image::imageops::resize(
            &cropped,
            target_w,
            target_h,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(resized)
    }

    /// Horizontal flip.
    #[cfg(feature = "augmentation")]
    pub fn horizontal_flip(&self, image: &RgbImage) -> RgbImage {
        image::imageops::flip_horizontal(image)
    }

    /// Color jitter: randomly adjust brightness, contrast, saturation, and hue.
    #[cfg(feature = "augmentation")]
    pub fn color_jitter(&mut self, image: &RgbImage) -> CnnResult<RgbImage> {
        let (width, height) = image.dimensions();
        let mut result = image.clone();

        // Sample jitter factors
        let brightness_factor = 1.0 + self.rng.gen_range(-self.config.brightness..=self.config.brightness);
        let contrast_factor = 1.0 + self.rng.gen_range(-self.config.contrast..=self.config.contrast);
        let saturation_factor = 1.0 + self.rng.gen_range(-self.config.saturation..=self.config.saturation);
        let hue_shift = self.rng.gen_range(-self.config.hue..=self.config.hue);

        // Compute image mean for contrast adjustment
        let mean = self.compute_mean(image);

        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y);
                let mut rgb = [pixel[0] as f64 / 255.0, pixel[1] as f64 / 255.0, pixel[2] as f64 / 255.0];

                // Apply brightness
                for c in rgb.iter_mut() {
                    *c *= brightness_factor;
                }

                // Apply contrast
                for (i, c) in rgb.iter_mut().enumerate() {
                    *c = (*c - mean[i]) * contrast_factor + mean[i];
                }

                // Apply saturation and hue in HSV space
                let (h, s, v) = rgb_to_hsv(rgb[0], rgb[1], rgb[2]);
                let new_s = (s * saturation_factor).clamp(0.0, 1.0);
                let new_h = (h + hue_shift * 360.0).rem_euclid(360.0);
                let (r, g, b) = hsv_to_rgb(new_h, new_s, v);

                rgb = [r, g, b];

                // Clamp and convert back to u8
                let out_pixel = Rgb([
                    (rgb[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (rgb[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (rgb[2] * 255.0).clamp(0.0, 255.0) as u8,
                ]);
                result.put_pixel(x, y, out_pixel);
            }
        }

        Ok(result)
    }

    /// Convert to grayscale (but keep 3 channels).
    #[cfg(feature = "augmentation")]
    pub fn to_grayscale(&self, image: &RgbImage) -> RgbImage {
        let (width, height) = image.dimensions();
        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y);
                // Luminance formula: 0.299*R + 0.587*G + 0.114*B
                let gray = (0.299 * pixel[0] as f64
                    + 0.587 * pixel[1] as f64
                    + 0.114 * pixel[2] as f64) as u8;
                result.put_pixel(x, y, Rgb([gray, gray, gray]));
            }
        }

        result
    }

    /// Gaussian blur (simplified box blur implementation).
    #[cfg(feature = "augmentation")]
    pub fn gaussian_blur(&mut self, image: &RgbImage) -> CnnResult<RgbImage> {
        let sigma = self.rng.gen_range(self.config.blur_sigma_range.0..=self.config.blur_sigma_range.1);

        // Use kernel size from config, or compute from sigma
        let kernel_size = if self.config.blur_kernel_size > 0 {
            self.config.blur_kernel_size
        } else {
            let k = (sigma * 6.0).ceil() as u32;
            if k % 2 == 0 { k + 1 } else { k }
        };

        // Generate Gaussian kernel
        let kernel = self.generate_gaussian_kernel(kernel_size, sigma);

        // Apply separable convolution
        let blurred = self.convolve_separable(image, &kernel)?;

        Ok(blurred)
    }

    /// Generate 1D Gaussian kernel.
    #[cfg(feature = "augmentation")]
    fn generate_gaussian_kernel(&self, size: u32, sigma: f64) -> Vec<f64> {
        let size = size as i32;
        let center = size / 2;
        let mut kernel = Vec::with_capacity(size as usize);
        let mut sum = 0.0;

        let sigma_sq_2 = 2.0 * sigma * sigma;

        for i in 0..size {
            let x = (i - center) as f64;
            let value = (-x * x / sigma_sq_2).exp();
            kernel.push(value);
            sum += value;
        }

        // Normalize
        for k in kernel.iter_mut() {
            *k /= sum;
        }

        kernel
    }

    /// Apply separable convolution (horizontal then vertical pass).
    #[cfg(feature = "augmentation")]
    fn convolve_separable(&self, image: &RgbImage, kernel: &[f64]) -> CnnResult<RgbImage> {
        let (width, height) = image.dimensions();
        let radius = kernel.len() / 2;

        // Horizontal pass
        let mut temp = ImageBuffer::<Rgb<u8>, _>::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0, 0.0, 0.0];
                for (i, &k) in kernel.iter().enumerate() {
                    let sx = (x as i32 + i as i32 - radius as i32).clamp(0, width as i32 - 1) as u32;
                    let pixel = image.get_pixel(sx, y);
                    sum[0] += pixel[0] as f64 * k;
                    sum[1] += pixel[1] as f64 * k;
                    sum[2] += pixel[2] as f64 * k;
                }
                temp.put_pixel(x, y, Rgb([
                    sum[0].clamp(0.0, 255.0) as u8,
                    sum[1].clamp(0.0, 255.0) as u8,
                    sum[2].clamp(0.0, 255.0) as u8,
                ]));
            }
        }

        // Vertical pass
        let mut result = ImageBuffer::<Rgb<u8>, _>::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let mut sum = [0.0, 0.0, 0.0];
                for (i, &k) in kernel.iter().enumerate() {
                    let sy = (y as i32 + i as i32 - radius as i32).clamp(0, height as i32 - 1) as u32;
                    let pixel = temp.get_pixel(x, sy);
                    sum[0] += pixel[0] as f64 * k;
                    sum[1] += pixel[1] as f64 * k;
                    sum[2] += pixel[2] as f64 * k;
                }
                result.put_pixel(x, y, Rgb([
                    sum[0].clamp(0.0, 255.0) as u8,
                    sum[1].clamp(0.0, 255.0) as u8,
                    sum[2].clamp(0.0, 255.0) as u8,
                ]));
            }
        }

        Ok(result)
    }

    /// Compute mean pixel value per channel.
    #[cfg(feature = "augmentation")]
    fn compute_mean(&self, image: &RgbImage) -> [f64; 3] {
        let (width, height) = image.dimensions();
        let n = (width * height) as f64;
        let mut sum = [0.0, 0.0, 0.0];

        for pixel in image.pixels() {
            sum[0] += pixel[0] as f64 / 255.0;
            sum[1] += pixel[1] as f64 / 255.0;
            sum[2] += pixel[2] as f64 / 255.0;
        }

        [sum[0] / n, sum[1] / n, sum[2] / n]
    }
}

impl Default for ContrastiveAugmentation {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// Convert RGB to HSV.
#[cfg(feature = "augmentation")]
fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;

    let s = if max > 1e-8 { delta / max } else { 0.0 };

    let h = if delta < 1e-8 {
        0.0
    } else if (max - r).abs() < 1e-8 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < 1e-8 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    (h, s, v)
}

/// Convert HSV to RGB.
#[cfg(feature = "augmentation")]
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());

    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = v - c;
    (r1 + m, g1 + m, b1 + m)
}

#[cfg(all(test, feature = "augmentation"))]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        let mut img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = 128;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        img
    }

    #[test]
    fn test_augmentation_builder() {
        let aug = ContrastiveAugmentation::builder()
            .crop_scale(0.5, 1.0)
            .horizontal_flip_prob(0.3)
            .output_size(128, 128)
            .seed(42)
            .build();

        assert_eq!(aug.config.crop_scale_min, 0.5);
        assert_eq!(aug.config.horizontal_flip_prob, 0.3);
        assert_eq!(aug.config.output_size, (128, 128));
    }

    #[test]
    fn test_random_resized_crop() {
        let mut aug = ContrastiveAugmentation::builder()
            .output_size(64, 64)
            .seed(42)
            .build();

        let img = create_test_image(256, 256);
        let cropped = aug.random_resized_crop(&img).unwrap();

        assert_eq!(cropped.dimensions(), (64, 64));
    }

    #[test]
    fn test_horizontal_flip() {
        let aug = ContrastiveAugmentation::default();
        let img = create_test_image(4, 4);
        let flipped = aug.horizontal_flip(&img);

        // Check that leftmost pixel is now rightmost
        assert_eq!(flipped.get_pixel(3, 0), img.get_pixel(0, 0));
        assert_eq!(flipped.get_pixel(0, 0), img.get_pixel(3, 0));
    }

    #[test]
    fn test_color_jitter() {
        let mut aug = ContrastiveAugmentation::builder()
            .color_jitter(0.2, 0.2, 0.2, 0.05)
            .seed(42)
            .build();

        let img = create_test_image(64, 64);
        let jittered = aug.color_jitter(&img).unwrap();

        // Should have same dimensions
        assert_eq!(jittered.dimensions(), img.dimensions());

        // Should be different from original (with high probability)
        let diff: u32 = img
            .pixels()
            .zip(jittered.pixels())
            .map(|(p1, p2)| {
                (p1[0] as i32 - p2[0] as i32).unsigned_abs()
                    + (p1[1] as i32 - p2[1] as i32).unsigned_abs()
                    + (p1[2] as i32 - p2[2] as i32).unsigned_abs()
            })
            .sum();
        assert!(diff > 0);
    }

    #[test]
    fn test_grayscale() {
        let aug = ContrastiveAugmentation::default();
        let img = create_test_image(64, 64);
        let gray = aug.to_grayscale(&img);

        // Check that all channels are equal
        for pixel in gray.pixels() {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
        }
    }

    #[test]
    fn test_gaussian_blur() {
        let mut aug = ContrastiveAugmentation::builder()
            .gaussian_blur(5, (1.0, 1.0))
            .seed(42)
            .build();

        let img = create_test_image(64, 64);
        let blurred = aug.gaussian_blur(&img).unwrap();

        assert_eq!(blurred.dimensions(), img.dimensions());
    }

    #[test]
    fn test_generate_pair() {
        let mut aug = ContrastiveAugmentation::builder()
            .output_size(32, 32)
            .seed(42)
            .build();

        let img = DynamicImage::ImageRgb8(create_test_image(128, 128));
        let (view1, view2) = aug.generate_pair(&img).unwrap();

        // Both views should have target size
        assert_eq!(view1.dimensions(), (32, 32));
        assert_eq!(view2.dimensions(), (32, 32));

        // Views should be different
        let diff: u32 = view1
            .pixels()
            .zip(view2.pixels())
            .map(|(p1, p2)| {
                (p1[0] as i32 - p2[0] as i32).unsigned_abs()
                    + (p1[1] as i32 - p2[1] as i32).unsigned_abs()
                    + (p1[2] as i32 - p2[2] as i32).unsigned_abs()
            })
            .sum();
        assert!(diff > 0, "Two augmented views should differ");
    }

    #[test]
    fn test_full_pipeline() {
        let mut aug = ContrastiveAugmentation::builder()
            .crop_scale(0.5, 1.0)
            .horizontal_flip_prob(1.0) // Always flip for testing
            .color_jitter(0.3, 0.3, 0.3, 0.1)
            .grayscale_prob(0.0) // Never grayscale for consistent test
            .output_size(48, 48)
            .seed(12345)
            .build();

        let img = DynamicImage::ImageRgb8(create_test_image(200, 200));
        let result = aug.augment(&img).unwrap();

        assert_eq!(result.dimensions(), (48, 48));
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let test_values = [
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Green
            (0.0, 0.0, 1.0), // Blue
            (0.5, 0.5, 0.5), // Gray
            (1.0, 1.0, 1.0), // White
            (0.0, 0.0, 0.0), // Black
        ];

        for (r, g, b) in test_values {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let (r2, g2, b2) = hsv_to_rgb(h, s, v);

            assert!((r - r2).abs() < 1e-6, "R mismatch for ({}, {}, {})", r, g, b);
            assert!((g - g2).abs() < 1e-6, "G mismatch for ({}, {}, {})", r, g, b);
            assert!((b - b2).abs() < 1e-6, "B mismatch for ({}, {}, {})", r, g, b);
        }
    }

    #[test]
    fn test_default_config() {
        let config = AugmentationConfig::default();

        assert!((config.crop_scale_min - 0.08).abs() < 1e-6);
        assert!((config.crop_scale_max - 1.0).abs() < 1e-6);
        assert!((config.horizontal_flip_prob - 0.5).abs() < 1e-6);
        assert_eq!(config.output_size, (224, 224));
    }
}

#[cfg(test)]
mod tests_no_feature {
    use super::*;

    #[test]
    fn test_builder_without_image_feature() {
        // This test should work even without the augmentation feature
        let aug = ContrastiveAugmentation::builder()
            .crop_scale(0.5, 1.0)
            .horizontal_flip_prob(0.3)
            .output_size(128, 128)
            .seed(42)
            .build();

        assert_eq!(aug.config().crop_scale_min, 0.5);
        assert_eq!(aug.config().horizontal_flip_prob, 0.3);
    }

    #[test]
    fn test_default_config() {
        let config = AugmentationConfig::default();
        assert!((config.crop_scale_min - 0.08).abs() < 1e-6);
        assert_eq!(config.output_size, (224, 224));
    }
}

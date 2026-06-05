use image::{self, ImageFormat};

use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupportedImageFormat {
    Png,
    Jpeg,
}

#[derive(Debug, Clone)]
pub struct ImageByteBuffer {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub format: SupportedImageFormat,
    pub bytes: Vec<u8>,
}

impl ImageByteBuffer {
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug)]
pub enum ImageIoError {
    Io(std::io::Error),
    Decode(image::ImageError),
    UnsupportedFormat(String),
}

impl fmt::Display for ImageIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageIoError::Io(err) => write!(f, "image I/O error: {}", err),
            ImageIoError::Decode(err) => write!(f, "image decode error: {}", err),
            ImageIoError::UnsupportedFormat(fmt_name) => {
                write!(
                    f,
                    "unsupported image format: {} (only png/jpg/jpeg)",
                    fmt_name
                )
            }
        }
    }
}

impl std::error::Error for ImageIoError {}

impl From<std::io::Error> for ImageIoError {
    fn from(value: std::io::Error) -> Self {
        ImageIoError::Io(value)
    }
}

impl From<image::ImageError> for ImageIoError {
    fn from(value: image::ImageError) -> Self {
        ImageIoError::Decode(value)
    }
}

fn format_from_image_format(format: ImageFormat) -> Option<SupportedImageFormat> {
    match format {
        ImageFormat::Png => Some(SupportedImageFormat::Png),
        ImageFormat::Jpeg => Some(SupportedImageFormat::Jpeg),
        _ => None,
    }
}

pub fn decode_png_or_jpeg_bytes(bytes: &[u8]) -> Result<ImageByteBuffer, ImageIoError> {
    let guessed = image::guess_format(bytes)?;
    let Some(format) = format_from_image_format(guessed) else {
        return Err(ImageIoError::UnsupportedFormat(format!("{:?}", guessed)));
    };

    let dynamic = image::load_from_memory_with_format(bytes, guessed)?;
    let grayscale = dynamic.to_luma8();

    Ok(ImageByteBuffer {
        width: grayscale.width(),
        height: grayscale.height(),
        channels: 1,
        format,
        bytes: grayscale.into_raw(),
    })
}

pub fn load_png_or_jpeg_from_path(path: &Path) -> Result<ImageByteBuffer, ImageIoError> {
    let bytes = fs::read(path)?;
    decode_png_or_jpeg_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::codecs::jpeg::JpegEncoder;
    use image::codecs::png::PngEncoder;
    use image::{ColorType, ImageEncoder, RgbImage};

    fn sample_rgb() -> RgbImage {
        RgbImage::from_fn(4, 4, |x, y| {
            image::Rgb([(x * 50) as u8, (y * 50) as u8, ((x + y) * 30) as u8])
        })
    }

    #[test]
    fn decodes_png_to_grayscale_buffer() {
        let rgb = sample_rgb();
        let mut encoded = Vec::new();

        {
            let encoder = PngEncoder::new(&mut encoded);
            encoder
                .write_image(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    ColorType::Rgb8.into(),
                )
                .expect("png encoding should succeed");
        }

        let buffer = decode_png_or_jpeg_bytes(&encoded).expect("png decode should succeed");

        assert_eq!(buffer.format, SupportedImageFormat::Png);
        assert_eq!(buffer.channels, 1);
        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 4);
        assert_eq!(buffer.bytes.len(), 16);
    }

    #[test]
    fn decodes_jpeg_to_grayscale_buffer() {
        let rgb = sample_rgb();
        let mut encoded = Vec::new();

        {
            let mut encoder = JpegEncoder::new_with_quality(&mut encoded, 90);
            encoder
                .encode(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    ColorType::Rgb8.into(),
                )
                .expect("jpeg encoding should succeed");
        }

        let buffer = decode_png_or_jpeg_bytes(&encoded).expect("jpeg decode should succeed");

        assert_eq!(buffer.format, SupportedImageFormat::Jpeg);
        assert_eq!(buffer.channels, 1);
        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 4);
        assert_eq!(buffer.bytes.len(), 16);
    }
}

use crate::helpers::image_dendrite::ImageDendrite;
use crate::helpers::image_io::{ImageByteBuffer, ImageIoError, load_png_or_jpeg_from_path};
use crate::helpers::legacy::image_controller::ImageNodeController;
use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::{NodeMetadata, NodeNetwork};
use crate::helpers::text_dendrite::DendriteType;

use std::path::Path;

pub type ImageNeuralNetwork = NeuralNetwork<ImageNodeController, ImageDendrite>;

impl NeuralNetwork<ImageNodeController, ImageDendrite> {
    pub fn new_image() -> Self {
        Self::with_controller(ImageNodeController)
    }

    pub fn enumerate_image_path(
        &self,
        image_bytes: &[u8],
    ) -> (Option<ImageDendrite>, Vec<ImageDendrite>) {
        self.enumerate_path_content(image_bytes)
    }

    pub fn enumerate_image_buffer_path(
        &self,
        image_buffer: &ImageByteBuffer,
    ) -> (Option<ImageDendrite>, Vec<ImageDendrite>) {
        self.enumerate_path_content(image_buffer.as_slice())
    }

    pub fn enumerate_image_path_from_file(
        &self,
        path: &Path,
    ) -> Result<(Option<ImageDendrite>, Vec<ImageDendrite>), ImageIoError> {
        let image_buffer = load_png_or_jpeg_from_path(path)?;
        Ok(self.enumerate_path_content(image_buffer.as_slice()))
    }

    pub fn insert_image(
        &mut self,
        image_bytes: &[u8],
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.insert_content(image_bytes, metadata, dendrite_type)
    }

    pub fn insert_image_buffer(
        &mut self,
        image_buffer: &ImageByteBuffer,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.insert_content(image_buffer.as_slice(), metadata, dendrite_type)
    }

    pub fn insert_image_from_file(
        &mut self,
        path: &Path,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) -> Result<(), ImageIoError> {
        let image_buffer = load_png_or_jpeg_from_path(path)?;
        self.insert_content(image_buffer.as_slice(), metadata, dendrite_type);
        Ok(())
    }
}

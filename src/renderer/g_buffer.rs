use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::AttachmentImage;

pub struct GBuffer {
    pub depth: Arc<AttachmentImage<Format>>,
}

impl GBuffer {
    pub fn new(device: &Arc<Device>, dimensions: [u32; 2]) -> Self {
        let depth = AttachmentImage::transient(
            device.clone(),
            dimensions,        //TODO abstract out
            Format::D32Sfloat, //TODO abstract out
        ).expect("failed to create depth buffer!");

        Self { depth }
    }
}

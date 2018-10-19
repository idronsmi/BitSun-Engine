use std::sync::Arc;
use vulkano::device::{Device, DeviceExtensions, DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract};

pub struct RenderpassManager {
    pub renderpass: Arc<RenderPassAbstract + Send + Sync>,
}

impl RenderpassManager {
    pub fn new(device: &Arc<Device>, swapchain_format: Format) -> Self {
        let renderpass = Arc::new(
            single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain_format,
                    samples: 1,
                },
                depth:{
                    load: Clear,
                    store: DontCare,
                    format: Format::D32Sfloat,
                    samples: 1,
                }
            },
            pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            ).unwrap(),
        );
        Self { renderpass }
    }

    pub fn get_render_pass(&self) -> Arc<RenderPassAbstract + Send + Sync> {
        self.renderpass.clone()
    }
    // fn find_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    // for (VkFormat format : candidates) {
    //     VkFormatProperties props;
    //     vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

    //     if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
    //         return format;
    //     } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
    //         return format;
    //     }
    // }
}

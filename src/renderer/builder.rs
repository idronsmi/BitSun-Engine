use renderer::g_buffer::GBuffer;
use renderer::queues::BsQueues;
use renderer::renderer::{Renderer, Texture};
use renderer::renderpass_manager::RenderpassManager;
use renderer::shaders::{fragment_shader, vertex_shader};
use renderer::window::Window;

use std::path::PathBuf;
use std::sync::Arc;

use image;

use vulkano;
use vulkano::buffer::{
    immutable::ImmutableBuffer, BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool,
    TypedBufferAccess,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::device::{ Features, Device, DeviceExtensions, DeviceOwned, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{
    swapchain::SwapchainImage, AttachmentImage, Dimensions::Dim2d, ImageLayout, ImageUsage,
    ImmutableImage, MipmapsCount,
};

use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::instance::{
    layers_list, Instance, InstanceExtensions, PhysicalDevice, PhysicalDevicesIter,
    QueueFamily,
};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{
    Capabilities, ColorSpace, CompositeAlpha, PresentMode, SupportedPresentModes, Swapchain,
};
use vulkano::sync::{self, GpuFuture};

use vulkano_win::required_extensions;

use winit;

const HEIGHT: u32 = 1024;
const WIDTH: u32 = 768;

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];
#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
    texCoord: [f32; 2],
}
impl Vertex {
    fn new(pos: [f32; 3], texCoord: [f32; 2]) -> Self {
        Self { pos, texCoord }
    }
}
impl_vertex!(Vertex, pos, texCoord);

fn vertices() -> [Vertex; 24] {
    [
        Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0]),
        Vertex::new([0.5, -0.5, 0.5], [1.0, 0.0]),
        Vertex::new([0.5, 0.5, 0.5], [1.0, 1.0]),
        Vertex::new([-0.5, 0.5, 0.5], [0.0, 1.0]),

        Vertex::new([-0.5, -0.5, -0.5], [0.0, 0.0]),
        Vertex::new([0.5, -0.5, -0.5], [1.0, 0.0]),
        Vertex::new([0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([-0.5, 0.5, -0.5], [0.0, 1.0]),

        Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0]),
        Vertex::new([0.5, -0.5, 0.5], [1.0, 0.0]),
        Vertex::new([0.5, -0.5, -0.5], [1.0, 1.0]),
        Vertex::new([-0.5, -0.5, -0.5], [0.0, 1.0]),

        Vertex::new([-0.5, 0.5, 0.5], [0.0, 0.0]),
        Vertex::new([0.5, 0.5, 0.5], [1.0, 0.0]),
        Vertex::new([0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([-0.5, 0.5, -0.5], [0.0, 1.0]),

        Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0]),
        Vertex::new([-0.5, 0.5, 0.5], [0.0, 1.0]),
        Vertex::new([-0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([-0.5, -0.5, -0.5], [1.0, 0.0]),

        Vertex::new([0.5, -0.5, 0.5], [0.0, 0.0]),
        Vertex::new([0.5, 0.5, 0.5], [0.0, 1.0]),
        Vertex::new([0.5, 0.5, -0.5], [1.0, 1.0]),
        Vertex::new([0.5, -0.5, -0.5], [1.0, 0.0]),
    ]
}

fn indices() -> [u16; 36] {
    [
        0,1,2,2,3,0,
        4,5,6,6,7,4,
        8,9,10,10,11,8,
        12,13,14,14,15,12,
        16,17,18,18,19,16,
        20,21,22,22,23,20
    ]
}

pub struct RendererBuilder {
    pub engine_settings: String,
    pub instance: Option<Arc<Instance>>,
}

impl RendererBuilder {
    pub fn new(engine_settings: &str) -> Self {
        let engine_settings = engine_settings.to_string();
        Self {
            engine_settings,
            instance: None,
        }
    }

    pub fn build(self, mut window: Window) -> Result<Renderer, String> {
        let instance = {
            match self.instance {
                Some(ref inst) => inst.clone(),
                None => return Err(String::from("Tried to build without instance!")),
            }
        };

        let debug_callback = Self::setup_debug_callback(&instance);
        let physical_device_temp = rank_devices(PhysicalDevice::enumerate(&instance));
        //uwnrap the device
        let physical_device = {
            match physical_device_temp {
                Some(device) => device,
                None => return Err("No physical device found!".to_string()),
            }
        };

        //Find needed queues
        let phys_queues = find_queues(&physical_device, &mut window);

        let minimal_features = Features {
            sampler_anisotropy: true,
            ..Features::none()
        };

        let (device, queues) = Device::new(
            physical_device,
            &minimal_features,
            &Self::device_extensions(),
            phys_queues,
        ).expect("failed to create logical device!");

        let queues = BsQueues::new(queues);
        let texture = Self::create_texture_image(&queues);

        let capabilities = window
            .surface()
            .capabilities(physical_device)
            .expect("failed to get surface capabilities");

        let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.is_some()
            && image_count > capabilities.max_image_count.unwrap()
        {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let (swap_chain, swap_chain_images) = Swapchain::new(
            device.clone(),
            window.surface().clone(),
            image_count,
            surface_format.0, // TODO: color space?
            extent,
            1, // layers
            image_usage,
            &queues.graphics,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            true, // clipped
            None,
        ).expect("failed to create swap chain!");

        let renderpass_manager = RenderpassManager::new(&device, swap_chain.format());
        let graphics_pipeline = Self::create_graphics_pipeline(
            &device,
            swap_chain.dimensions(),
            renderpass_manager.get_render_pass(),
        );
        let g_buffer = GBuffer::new(&device, extent.clone());

        let swap_chain_framebuffers = Self::create_framebuffers(
            &swap_chain_images,
            g_buffer.depth.clone(),
            renderpass_manager.get_render_pass(),
        );
        let uniform_buffer_pool = CpuBufferPool::<vertex_shader::ty::UniformBufferObject>::new(
            device.clone(),
            BufferUsage::all(),
        );

        let vertex_buffer = Self::create_vertex_buffer(&queues.graphics);
        let index_buffer = Self::create_index_buffer(&queues.graphics);

        let previous_frame_end = Some(Self::create_sync_objects(&device));

        let renderer = Renderer::build_from_builder(
            window,
            device,
            queues,
            texture,
            swap_chain,
            swap_chain_images,
            renderpass_manager,
            graphics_pipeline,
            swap_chain_framebuffers,
            uniform_buffer_pool,
            vertex_buffer,
            index_buffer,
            g_buffer,
            vec![],
            previous_frame_end,
            false,
        );

        Ok(renderer)
    }

    pub fn create_instance(&mut self) -> Result<(), String> {
        let extensions = Self::get_required_extensions();
        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            let instance = Instance::new(None, &extensions, VALIDATION_LAYERS.iter().map(|s| *s))
                .expect("failed to create Vulkan instance");
            self.instance = Some(instance);
            Ok({})
        } else {
            let instance =
                Instance::new(None, &extensions, None).expect("failed to create Vulkan instance");
            self.instance = Some(instance);
            Ok({})
        }
    }

    ///Returns an instance if there is already one, or takes the current information of the builder
    /// to create one and returns this instead.
    /// #Panic If this doesn't work it will panic.
    pub fn get_instance(&mut self) -> Arc<Instance> {
        match self.instance {
            Some(ref inst) => inst.clone(),
            None => {
                match self.create_instance() {
                    Ok(_) => {}
                    Err(_) => panic!("Failed to create an instance"),
                }
                //now return the instance which should be there now.
                self.instance
                    .clone()
                    .expect("there was no instance, but there should be one!")
            }
        }
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = Self::device_extensions();
        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn choose_swap_surface_format(
        available_formats: &[(Format, ColorSpace)],
    ) -> (Format, ColorSpace) {
        // NOTE: the 'preferred format' mentioned in the tutorial doesn't seem to be
        // queryable in Vulkano (no VK_FORMAT_UNDEFINED enum)
        *available_formats
            .iter()
            .find(|(format, color_space)| {
                println!("found {:?}, {:?}", format, color_space);
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            }).unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            return current_extent;
        } else {
            let mut actual_extent = [WIDTH, HEIGHT];
            actual_extent[0] = capabilities.min_image_extent[0]
                .max(capabilities.max_image_extent[0].min(actual_extent[0]));
            actual_extent[1] = capabilities.min_image_extent[1]
                .max(capabilities.max_image_extent[1].min(actual_extent[1]));
            actual_extent
        }
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: Arc<RenderPassAbstract + Send + Sync>,
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
        let vert_shader_module = vertex_shader::Shader::load(device.clone())
            .expect("failed to create vertex shader module!");
        let frag_shader_module = fragment_shader::Shader::load(device.clone())
            .expect("failed to create fragment shader module!");

        let dimensions = [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        Arc::new(
            GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(frag_shader_module.main_entry_point(), ())
            .depth_clamp(false)
            .depth_stencil_simple_depth()
            // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
            .polygon_mode_fill() // = default
            .line_width(1.0) // = default
            .front_face_clockwise()
            // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
            .blend_pass_through() // = default
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device.clone())
            .unwrap(),
        )
    }

    fn create_framebuffers(
        swap_chain_images: &Vec<Arc<SwapchainImage<winit::Window>>>,
        depth_buffer: Arc<AttachmentImage<Format>>,
        render_pass: Arc<RenderPassAbstract + Send + Sync>,
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        swap_chain_images
            .iter()
            .map(|image| {
                let fba: Arc<FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(image.clone())
                        .unwrap()
                        .add(depth_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                fba
            }).collect::<Vec<_>>()
    }

    fn create_vertex_buffer(graphics_queue: &Arc<Queue>) -> Arc<BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices().iter().cloned(),
            BufferUsage::vertex_buffer(),
            graphics_queue.clone(),
        ).unwrap();
        future.flush().unwrap();
        buffer
    }

    fn create_index_buffer(
        graphics_queue: &Arc<Queue>,
    ) -> Arc<TypedBufferAccess<Content = [u16]> + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices().iter().cloned(),
            BufferUsage::index_buffer(),
            graphics_queue.clone(),
        ).unwrap();
        future.flush().unwrap();
        buffer
    }

    pub fn create_texture_image(queues: &BsQueues) -> Arc<Texture> {
        let image =
            image::open(PathBuf::from("./assets/textures/statue.jpg")).expect("bad image path");
        let image = image.to_rgba();
        let (width, height) = image.dimensions();
        let image_dimensions = Dim2d {
            width: width,
            height: height,
        };
        let image_data = image.into_raw().clone();
        let image_format = vulkano::format::Format::R8G8B8A8Unorm;

        let transfer_buffer = CpuAccessibleBuffer::from_iter(
            queues.graphics.device().clone(),
            BufferUsage::transfer_source(),
            image_data.iter().cloned(),
        ).expect("Failed to generate CPU Buffer for image creation!");

        let usage = ImageUsage {
            transfer_destination: true,
            transfer_source: true,
            sampled: true,
            ..ImageUsage::none()
        };
        let layout = ImageLayout::General;

        let (img, init) = ImmutableImage::uninitialized(
            transfer_buffer.device().clone(),
            image_dimensions,
            image_format,
            MipmapsCount::One,
            usage,
            layout,
            transfer_buffer.device().active_queue_families(),
        ).expect("failed to create uninitialized image!");

        let arc_init = Arc::new(init);

        let device = transfer_buffer.device().clone();
        let mut command_buffer =
            AutoCommandBufferBuilder::new(device.clone(), queues.graphics.family())
                .expect("Failed to start command buffer for image creation!");
        command_buffer = command_buffer
            .copy_buffer_to_image_dimensions(
                transfer_buffer,
                arc_init,
                [0, 0, 0],
                image_dimensions.width_height_depth(),
                0,
                1,
                0,
            ).expect("failed to copy initial image to image_buffer!");

        let final_command_buffer = command_buffer
            .build()
            .expect("failed to build image command_buffer");

        let future = final_command_buffer
            .execute(queues.graphics.clone())
            .expect("Image Gen Error");

        let after_future = future
            .then_signal_fence_and_flush()
            .expect("failed to flush texture upload");

        after_future
            .wait(None)
            .expect("failed to wait for texture upload and blit");

        let mip_mapping_levels = img.mipmap_levels() as f32;
        //let (mip_min, mip_max) = (0.0, mip_mapping_levels as f32);
        //TODO Remove when uploading error is resolved within vulkano
        let (mip_min, mip_max) = (0.0, 0.0);

        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            16.0,
            mip_min,
            mip_max,
        ).expect("couldnt make sampler");

        let temp_texture = Texture {
            texture: img,
            sampler: sampler,
        };

        Arc::new(temp_texture)
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<GpuFuture>
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = required_extensions();
        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_report = true;
        }

        extensions
    }

    fn device_extensions() -> DeviceExtensions {
        DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list()
            .unwrap()
            .map(|l| l.name().to_owned())
            .collect();
        VALIDATION_LAYERS
            .iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let msg_types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: false,
            debug: true,
        };
        DebugCallback::new(&instance, msg_types, |msg| {
            println!("validation layer: {:?}", msg.description);
        }).ok()
    }
}

///Analyses the availabe queues on the device and returns an apropriate weighted array, used to build
/// the artificial devices and queues.
fn find_queues<'a>(
    physical_device: &PhysicalDevice<'a>,
    window: &mut Window,
) -> Vec<(QueueFamily<'a>, f32)> {
    println!("QUEUEINFO:\n==========",);
    //Create a queue
    for queue in physical_device.queue_families() {
        print!(
            "Queue {}, graph: {}, comp: {}, count: {}",
            queue.id(),
            queue.supports_graphics(),
            queue.supports_compute(),
            queue.queues_count()
        );
    }
    println!("==========",);

    let mut queue_collection = Vec::new();

    //After showing them for debug resasons, try to classify them, first find a presenter queue, if
    //thats not possible, panic, since we want to show something. Then try to find a compute and transfer
    //queue, if not possible, just let them be, the final artificial queues will be cloned correctly.
    let mut has_presenter = false;
    let mut has_compute = false;
    let mut has_transfer = false;
    for queue in physical_device.queue_families() {
        //Check for the graphics queue, which is always needed
        if queue.supports_graphics()
            && window.surface().is_supported(queue).unwrap_or(false)
            && !has_presenter
        {
            //We have a graphics queue, push with highest priority
            queue_collection.push((queue, 1.0));
            has_presenter = true
        }
        //If we have already a graphics queue, check for a compute capable queue for async compute
        //without graphics capablilitys
        if queue.supports_compute() && !queue.supports_graphics() && !has_compute {
            queue_collection.push((queue, 0.75));
            has_compute = true;
        }

        //Finally check for one which can only upload. If there is one, use it as upload queue
        if !queue.supports_compute() && !queue.supports_graphics() && !has_transfer {
            queue_collection.push((queue, 0.5));
            has_transfer = true;
        }
    }

    println!(
        "Found queues: graphics: {}, compute: {}, transfer: {}",
        has_presenter, has_compute, has_transfer
    );

    queue_collection
}

///A function to rank a iterator of physical devices. The best one will be returned
fn rank_devices(devices: PhysicalDevicesIter) -> Option<PhysicalDevice> {
    use std::collections::BTreeMap;
    use vulkano::instance::PhysicalDeviceType;
    //save the devices according to the score, at the end pick the last one (highest score);
    let mut ranking = BTreeMap::new();

    for device in devices.into_iter() {
        let mut device_score = 0;

        match device.ty() {
            PhysicalDeviceType::IntegratedGpu => device_score += 10,
            PhysicalDeviceType::DiscreteGpu => device_score += 50,
            PhysicalDeviceType::VirtualGpu => device_score += 20,
            PhysicalDeviceType::Cpu => device_score += 5,
            PhysicalDeviceType::Other => device_score += 0,
        }

        ranking.insert(device_score, device);
    }

    let mut tmp_vec = Vec::new();
    for (_, device) in ranking.into_iter().rev() {
        tmp_vec.push(device);
    }

    if tmp_vec.len() > 0 {
        Some(tmp_vec[0])
    } else {
        None
    }
}

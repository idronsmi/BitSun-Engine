use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use renderer::g_buffer::GBuffer;
use renderer::queues::BsQueues;
use renderer::renderpass_manager::RenderpassManager;
use renderer::shaders;
use renderer::window::Window;

use cgmath::{perspective, Deg, Matrix4, Point3, Rad, Vector3};

use vulkano;
use vulkano::buffer::{
    BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess,
};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceOwned};

use vulkano::framebuffer::{FramebufferAbstract, RenderPassAbstract};
use vulkano::image::{swapchain::SwapchainImage, ImmutableImage};
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{acquire_next_image, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, GpuFuture};

use winit;

pub struct Texture {
    pub sampler: Arc<Sampler>,
    pub texture: Arc<ImmutableImage<vulkano::format::Format>>,
}

#[allow(unused)]
pub struct Renderer {
    window: Window,
    device: Arc<Device>,
    queues: BsQueues,
    texture: Arc<Texture>,
    swap_chain: Arc<Swapchain<winit::Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<winit::Window>>>,
    renderpass_manager: RenderpassManager,
    graphics_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    swap_chain_framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    uniform_buffer_pool: CpuBufferPool<shaders::vertex_shader::ty::UniformBufferObject>,
    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>,
    g_buffer: GBuffer,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,
    previous_frame_end: Option<Box<GpuFuture>>,
    recreate_swap_chain: bool,
    start_time: Instant,
}

impl Renderer {
    pub fn build_from_builder(
        window: Window,
        device: Arc<Device>,
        queues: BsQueues,
        texture: Arc<Texture>,
        swap_chain: Arc<Swapchain<winit::Window>>,
        swap_chain_images: Vec<Arc<SwapchainImage<winit::Window>>>,
        renderpass_manager: RenderpassManager,
        graphics_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
        swap_chain_framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
        uniform_buffer_pool: CpuBufferPool<shaders::vertex_shader::ty::UniformBufferObject>,
        vertex_buffer: Arc<BufferAccess + Send + Sync>,
        index_buffer: Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>,
        g_buffer: GBuffer,
        command_buffers: Vec<Arc<AutoCommandBuffer>>,
        previous_frame_end: Option<Box<GpuFuture>>,
        recreate_swap_chain: bool,
    ) -> Self {
        let start_time = Instant::now();
        Self {
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
            command_buffers,
            previous_frame_end,
            recreate_swap_chain,
            start_time,
        }
    }

    #[allow(unused)]
    pub fn run(&mut self, mut events_loop: winit::EventsLoop) {
        let mut running = true;
        while running {
            events_loop.poll_events(|event| match event {
                winit::Event::WindowEvent {
                    event: winit::WindowEvent::CloseRequested,
                    ..
                } => running = false,
                _ => (),
            });

            self.draw_frame();
        }
    }

    fn update_uniform_buffer(&self) -> shaders::vertex_shader::ty::UniformBufferObject {
        let elapsed = self.start_time.elapsed();
        let time = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
        let temp = Deg(-55.0) * time as f32;

        let _model = Matrix4::from_angle_z(temp);
        let _view = Matrix4::look_at(
            Point3::new(3.0, 3.0, 3.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -3.0),
        );
        let _proj = perspective(
            Deg(45.0),
            self.swap_chain.dimensions()[0] as f32 / self.swap_chain.dimensions()[1] as f32,
            0.1,
            100.0,
        );

        let temp = shaders::vertex_shader::ty::UniformBufferObject {
            model: _model.into(),
            view: _view.into(),
            proj: _proj.into(),
        };

        temp
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, acquire_future) = match acquire_next_image(self.swap_chain.clone(), None)
        {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swap_chain = true;
                return;
            }
            Err(err) => panic!("{:?}", err),
        };
        self.create_command_buffers();
        let command_buffer = self.command_buffers[image_index].clone();
        let frame = self.swap_chain_framebuffers[image_index].clone();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queues.graphics.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queues.graphics.clone(),
                self.swap_chain.clone(),
                image_index,
            ).then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    pub fn recreate_swap_chain(&mut self) -> bool {
        //TODO new dimmensions etc

        let dimensions = self
            .window
            .surface()
            .capabilities(self.device.physical_device())
            .expect("failed to get surface capabilities")
            .current_extent
            .unwrap_or([1024, 768]);

        let (new_swapchain, new_images) = match self.swap_chain.recreate_with_dimension(dimensions)
        {
            Ok(r) => r,
            // This error tends to happen when the user is manually resizing the window.
            // Simply restarting the loop is the easiest way to fix this issue.
            Err(SwapchainCreationError::UnsupportedDimensions) => {
                return false;
            }
            Err(err) => panic!("{:?}", err),
        };

        mem::replace(&mut self.swap_chain, new_swapchain);
        mem::replace(&mut self.swap_chain_images, new_images);

        //with the new dimensions set in the setting, recreate the images of the frame system as well
        //self.frame_system.recreate_attachments();

        //Now when can mark the swapchain as "fine" again
        self.recreate_swap_chain = false;
        true
    }

    pub fn create_command_buffers(&mut self) {
       

        let queue_family = self.queues.graphics.family();
        let uniform_buffer_pool_subuffer = self
            .uniform_buffer_pool
            .next(self.update_uniform_buffer())
            .unwrap();

        let set_1 = Arc::new(
            PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 0)
                .add_buffer(uniform_buffer_pool_subuffer.clone())
                .expect("failed to add uniform buffer")
                .build()
                .expect("failed to build set 1"),
        );

        let set_2 = Arc::new(
            PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 1)
                .add_sampled_image(self.texture.texture.clone(), self.texture.sampler.clone())
                .expect("failed to add sampled image")
                .build()
                .expect("failed to build set 2"),
        );

        self.command_buffers = self
            .swap_chain_framebuffers
            .iter()
            .map(|framebuffer| {
                Arc::new(
                    AutoCommandBufferBuilder::primary_simultaneous_use(
                        self.device.clone(),
                        queue_family,
                    ).unwrap()
                    .begin_render_pass(
                        framebuffer.clone(),
                        false,
                        vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()],
                    ).unwrap()
                    .draw_indexed(
                        self.graphics_pipeline.clone(),
                        &DynamicState::none(),
                        vec![self.vertex_buffer.clone()],
                        self.index_buffer.clone(),
                        (set_1.clone(), set_2.clone()),
                        (),
                    ).unwrap()
                    .end_render_pass()
                    .unwrap()
                    .build()
                    .unwrap(),
                )
            }).collect();
    }

    // pub fn depth(
    //     &self,
    //     command_buffer: Arc<AutoCommandBuffer>,
    //     frame_buffer: Arc<FramebufferAbstract + Send + Sync>,
    // ) {
    //      let screen_sampler = Sampler::new(
    //         self.device.clone(),
    //         Filter::Linear,
    //         Filter::Linear,
    //         MipmapMode::Linear,
    //         SamplerAddressMode::ClampToEdge,
    //         SamplerAddressMode::ClampToEdge,
    //         SamplerAddressMode::ClampToEdge,
    //         0.0,
    //         1.0,
    //         1.0,
    //         1.0,
    //     ).expect("failed to create screen sampler");

    //     let clearings = vec![
    //         [0.0, 0.0, 0.0, 0.0].into()
    //     ];

    //     let mut new_cb = command_buffer
    //         .begin_render_pass(frame_buffer, false, clearings)
    //         .expect("failed to start assemble pass");

    //     PersistentDescriptorSet::start(self.pipeline.get_pipeline_ref(), 0)
    // }
}

use std::sync::Arc;

use vulkano::swapchain::Surface;
use vulkano::instance::Instance;


use vulkano_win::VkSurfaceBuild;

use winit;

pub const WIDTH: u32 = 1024;
pub const HEIGHT: u32 = 768;

pub struct Window {
    surface:  Arc<Surface<winit::Window>>,
}

impl Window {

    pub fn new(events_loop: &winit::EventsLoop, instance: &Arc<Instance>) -> Self {
        let builder = winit::WindowBuilder::new();

        let surface = builder
            .with_title("Test")
            .with_dimensions(winit::dpi::LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
            .build_vk_surface(events_loop, instance.clone())
            .unwrap();

        Self {
            surface
        }
    }

    ///Returns the window surface
    #[inline]
    pub fn surface(&mut self) -> &Arc<Surface<winit::Window>> {
        &self.surface
    }

    ///Returns the window component
    #[inline]
    pub fn window(&mut self) -> &winit::Window{
        &self.surface.window()
    }
}
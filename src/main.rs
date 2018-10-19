extern crate cgmath;
extern crate image;
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;
#[macro_use]
extern crate vulkano_shader_derive;

mod renderer;

use renderer::builder::RendererBuilder;

fn main() {
  let mut builder = RendererBuilder::new("");
  let events_loop = winit::EventsLoop::new();
  builder.create_instance().expect("bad instance");
  let instance = builder.get_instance();
  let window = renderer::window::Window::new(&events_loop, &instance);
  let mut renderer = builder.build(window).unwrap();
  renderer.run(events_loop);
}

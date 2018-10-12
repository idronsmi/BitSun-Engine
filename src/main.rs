#![windows_subsystem = "windows"]

#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;
#[macro_use]
extern crate vulkano_shader_derive;


mod renderer;
mod application;

use application::Application;

fn main() {
  let mut app = Application::new();
  app.run();
}

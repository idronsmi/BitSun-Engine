#[allow(unused)]
pub mod vertex_shader {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[path = "assets/shaders/triangle.vert"]
    struct Dummy;
}

#[allow(unused)]
pub mod fragment_shader {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[path = "assets/shaders/triangle.frag"]
    struct Dummy;
}

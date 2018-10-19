use renderer::shaders::vertex_shader;

use std::sync::Arc;

use cgmath::Matrix4;
use vulkano::buffer::{
    cpu_pool::{CpuBufferPool, CpuBufferPoolSubbuffer},
    BufferUsage,
};
use vulkano::device::Device;
use vulkano::memory::pool::StdMemoryPool;

pub struct UniformManager {
    data: vertex_shader::ty::UniformBufferObject,
    uniform_pool: CpuBufferPool<vertex_shader::ty::UniformBufferObject>,
}

impl UniformManager {
    pub fn new(device: Arc<Device>) -> Self {
        let data = vertex_shader::ty::UniformBufferObject {
            model: [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            proj: [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            view: [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        };

        //Create some pools to allocate from
        let buffer_pool = CpuBufferPool::<vertex_shader::ty::UniformBufferObject>::new(
            device.clone(),
            BufferUsage::all(),
        );

        Self {
            data: data,
            ///First uniform buffer pool block, used or model, view and perspecive matrix
            uniform_pool: buffer_pool,
        }
    }

    ///Returns a subbuffer of the u_world item, can be used to create a u_world_set
    pub fn get_subbuffer_data(
        &mut self,
        transform_matrix: Matrix4<f32>,
    ) -> CpuBufferPoolSubbuffer<vertex_shader::ty::UniformBufferObject, Arc<StdMemoryPool>>
    {
        //prepare the Data struct
        let mut tmp_data_struct = self.data.clone();

        tmp_data_struct.model = transform_matrix.into();

        match self.uniform_pool.next(tmp_data_struct) {
            Ok(k) => k,
            Err(e) => {
                println!("{:?}", e);
                panic!("failed to allocate new sub buffer!")
            }
        }
    }

    ///Updates the internal data used for the uniform buffer creation
    pub fn update(&mut self, new_data: vertex_shader::ty::UniformBufferObject) {
        self.data = new_data;
    }
}

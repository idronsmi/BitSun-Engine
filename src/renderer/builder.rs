use vulkano::device::{Device, DeviceExtensions,};
use vulkano::instance::{
    layers_list, Features, Instance, InstanceExtensions, PhysicalDevice, PhysicalDeviceType,QueueFamily 
};

use winit::Window;

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
fn rank_devices(
    devices: vulkano::instance::PhysicalDevicesIter,
) -> Option<vulkano::instance::PhysicalDevice> {
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

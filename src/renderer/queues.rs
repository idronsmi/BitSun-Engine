use std::sync::Arc;
use vulkano::device::{Queue, QueuesIter};
use vulkano::instance::QueueFamily;

///Collects all queues used on the gpu.
///The following ist guranteed:
/// - graphics: Can do graphics operation
/// - compute: Can do compute operation
/// - transfer: Can upload / Download to the gpu, but no operations.
/// However, all of thoose queues can probably do more, depending on the gpu type.
#[derive(Clone)]
pub struct BsQueues {
    pub graphics: Arc<Queue>,
    pub compute: Arc<Queue>,
    pub transfer: Arc<Queue>,
}

impl BsQueues {
    pub fn new(queues: QueuesIter) -> Self {
        let mut graphic_queue = None;
        let mut compute_queue = None;
        let mut transfer_queue = None;

        //Now check the queues which where created
        for q in queues {
            if q.family().supports_graphics() && graphic_queue.is_none() {
                graphic_queue = Some(q);
                continue;
            }

            if q.family().supports_compute() && compute_queue.is_none() {
                compute_queue = Some(q);
                continue;
            }

            if !q.family().supports_compute()
                && !q.family().supports_graphics()
                && transfer_queue.is_none()
            {
                transfer_queue = Some(q);
                continue;
            }
        }

        //Check the queues
        if graphic_queue.is_none() {
            panic!("No graphics queue found");
        }

        if compute_queue.is_none() {
            println!("WARNING: No compute queue found, using graphics queue",);
            compute_queue = graphic_queue.clone();
        }

        if transfer_queue.is_none() {
            println!("WARNING: No transfer queue found, using compute",);
            transfer_queue = compute_queue.clone();
        }

        BsQueues {
            graphics: graphic_queue.expect("Failed to find graphics queue"),
            compute: compute_queue.expect("Failed to find compute queue"),
            transfer: transfer_queue.expect("Failed to find transfer queue"),
        }
    }

    pub fn get_families(&self) -> Vec<QueueFamily<'_>> {
        vec![
            self.graphics.family(),
            self.compute.family(),
            self.transfer.family(),
        ]
    }
}

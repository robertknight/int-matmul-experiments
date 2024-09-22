use crate::Kernel;

#[cfg(target_arch = "x86_64")]
mod x86_64;

pub fn new_kernel() -> Box<dyn Kernel> {
    #[cfg(target_arch = "x86_64")]
    {
        Box::new(x86_64::AvxKernel::new().unwrap())
    }
}

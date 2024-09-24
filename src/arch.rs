use crate::Kernel;

#[cfg(target_arch = "x86_64")]
mod x86_64;

pub fn new_kernel() -> Box<dyn Kernel> {
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "avx512")]
    {
        use x86_64::avx512::Avx512Kernel;
        if let Some(kernel) = Avx512Kernel::new() {
            return Box::new(kernel);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(kernel) = x86_64::AvxKernel::new() {
            return Box::new(kernel);
        }
    }

    panic!("no supported kernel available");
}

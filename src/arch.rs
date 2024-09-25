use crate::Kernel;

mod generic;

#[cfg(target_arch = "x86_64")]
mod x86_64;

pub enum KernelHint {
    Generic,
    Avx512,
    Avx,
}

pub fn new_kernel(hint: Option<KernelHint>) -> Box<dyn Kernel> {
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "avx512")]
    {
        if matches!(hint, Some(KernelHint::Avx512) | None) {
            use x86_64::avx512::Avx512Kernel;
            if let Some(kernel) = Avx512Kernel::new() {
                return Box::new(kernel);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if matches!(hint, Some(KernelHint::Avx) | None) {
            if let Some(kernel) = x86_64::AvxKernel::new() {
                return Box::new(kernel);
            }
        }
    }

    Box::new(generic::GenericKernel::new())
}

use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use rand::Rng;

const PTX_SRC: &str = r#"
// Normal cumulative distribution function (CNDF)
extern "C" __device__ double normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

// Black-Scholes kernel running on the GPU
extern "C" __global__ void black_scholes_kernel(double *d_S, double *d_K, double *d_T,
                                     double r, double sigma,
                                     double *d_call, double *d_put, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        double S = d_S[idx];
        double K = d_K[idx];
        double T = d_T[idx];

        double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);

        d_call[idx] = S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2);
        d_put[idx] = K * exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);
    }
}

"#;

fn main() -> Result<(), DriverError>{
    let num_elements: u32 = 1_000_000;
    let mut rng = rand::rng();

    // Initialize the CUDA supported device
    let device = CudaDevice::new(0)?;
    println!("{:?}", device.name());

    // Initialize the required parameters on the host side

    // Immutable vectors (inputs)
    let s_host: Vec<f32> = (0..num_elements)
        .map(|_| rng.random_range(50.0..5000.0))
        .collect(); //Stock Prices
    let k_host: Vec<f32> = (0..num_elements)
        .map(|_| rng.random_range(50.0..5000.0))
        .collect(); // Strike Prices
    let t_host: Vec<f32> = (0..num_elements)
        .map(|_| rng.random_range(0.01..2.0))
        .collect(); // Time of expiry(in years)
    let r_host: Vec<f32> = (0..num_elements)
        .map(|_| rng.random_range(0.01..0.07))
        .collect(); // Risk-free interest rates
    let sigma_host: Vec<f32> = (0..num_elements)
        .map(|_| rng.random_range(0.1..0.90))
        .collect(); // Volatilities

    // Allocate buffers on the device
    let s_device = device.htod_copy(s_host)?;
    let k_device = device.htod_copy(k_host)?;
    let t_device = device.htod_copy(t_host)?;
    let r_device = device.htod_copy(r_host)?;
    let sigma_device = device.htod_copy(sigma_host)?;

    let mut c_device = device.alloc_zeros::<f32>(num_elements as usize)?;
    let mut p_device = device.alloc_zeros::<f32>(num_elements as usize)?;

    let start = std::time::Instant::now();

    let black_scholes_ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());
    device.load_ptx(black_scholes_ptx, "black-scholes", &["black_scholes_kernel"])?;

    let black_scholes_kernel = device.get_func("black-scholes", "black_scholes_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(num_elements);

    let kernel_start = std::time::Instant::now();
    unsafe {
        black_scholes_kernel.launch(
            cfg,
            (
                &s_device,
                &k_device,
                &t_device,
                &r_device,
                &sigma_device,
                &mut c_device,
                &mut p_device,
                num_elements as usize,
            ),
        )
    }?;
    // device.synchronize();
    println!("Kernel executed in {:?}", kernel_start.elapsed());

    let c_host: Vec<f32> = device.dtoh_sync_copy(&c_device)?;
    let p_host: Vec<f32> = device.dtoh_sync_copy(&p_device)?;

    Ok(())
}

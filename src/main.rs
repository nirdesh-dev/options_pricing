use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::E;

fn black_scholes(
    s0: f64,  // Stock price
    k: f64,   // Strike price
    r: f64,   // Risk-free rate
    t: f64,   // Time to expiration (years)
    sigma: f64, // Volatility
) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let d1 = (f64::ln(s0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let call_price = s0 * normal.cdf(d1) - k * (E.powf(-r * t)) * normal.cdf(d2);
    let put_price = k * (E.powf(-r * t)) * normal.cdf(-d2) - s0 * normal.cdf(-d1);

    (call_price, put_price)
}

fn main() {
    let s0 = 100.0;   // Current stock price
    let k = 100.0;    // Strike price
    let r = 0.05;     // Risk-free interest rate (5%)
    let t = 1.0;      // Time to expiration (1 year)
    let sigma = 0.2;  // Volatility (20%)

    let (call_price, put_price) = black_scholes(s0, k, r, t, sigma);

    println!("European Call Price: {:.2}", call_price);
    println!("European Put Price: {:.2}", put_price);
}
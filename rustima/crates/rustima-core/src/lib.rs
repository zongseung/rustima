pub mod batch;
pub mod css;
pub mod error;
pub mod forecast;
pub mod inference;
pub mod initialization;
pub mod kalman;
pub mod optimizer;
pub mod params;
pub mod pipeline;
pub mod polynomial;
pub mod score;
pub mod start_params;
pub mod state_space;
pub mod types;
pub mod validation;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
#[cfg(test)]
pub(crate) mod test_helpers;

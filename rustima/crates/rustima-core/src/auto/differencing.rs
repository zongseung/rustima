use nalgebra::{DMatrix, DVector};

use crate::error::{Result, SarimaxError};

fn difference(y: &[f64]) -> Vec<f64> {
    y.windows(2).map(|w| w[1] - w[0]).collect()
}

// ADF with a constant: compare AIC on a common sample, then refit the selected lag.
// This matches the former adfuller(maxlag=min(10, n/3), regression="c", autolag="AIC").
fn regression(y: &[f64], lag: usize, trim: usize) -> Result<(f64, f64)> {
    let n = y.len() - trim - 1;
    let k = lag + 2;
    let x = DMatrix::from_fn(n, k, |i, j| {
        let t = trim + 1 + i;
        match j {
            0 => y[t - 1],
            1 => 1.0,
            _ => y[t - j + 1] - y[t - j],
        }
    });
    let target = DVector::from_fn(n, |i, _| y[trim + 1 + i] - y[trim + i]);
    let svd = x.clone().svd(true, true);
    let tolerance = svd.singular_values.max() * 1e-15;
    let rank = svd
        .singular_values
        .iter()
        .filter(|&&v| v > tolerance)
        .count();
    let pinv = svd
        .pseudo_inverse(tolerance)
        .map_err(|e| SarimaxError::InvalidInput(format!("ADF regression: {e}")))?;
    let beta = &pinv * &target;
    let rss = (target - x * &beta).norm_squared();
    let variance = rss / (n - rank) as f64;
    let se = (variance * pinv.row(0).norm_squared()).sqrt();
    Ok((
        n as f64 * (rss / n as f64).ln() + 2.0 * rank as f64,
        beta[0] / se,
    ))
}

pub(super) fn adf_statistic(y: &[f64]) -> Result<(f64, usize)> {
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    let scaled: Vec<_> = y.iter().map(|v| v / scale).collect();
    let max_lag = 10.min(y.len() / 3);
    let mut best = (f64::INFINITY, 0);
    for lag in 0..=max_lag {
        let (aic, _) = regression(&scaled, lag, max_lag)?;
        if aic < best.0 {
            best = (aic, lag);
        }
    }
    Ok((regression(&scaled, best.1, best.1)?.1, best.1))
}

pub(super) fn ndiffs(y: &[f64], max_d: usize) -> Result<usize> {
    let mut current = y.to_vec();
    for d in 0..max_d {
        if current.len() < 10 || current.iter().all(|v| *v == current[0]) {
            return Ok(d);
        }
        // MacKinnon (1994), constant regression, N=1, p=0.05.
        // Root of Phi(2.1659 + 1.4412*t + 0.038269*t*t) = 0.05.
        // Reference: statsmodels v0.14.6, tsa/adfvalues.py (tau_c_smallp).
        if adf_statistic(&current)?.0 < -2.861_592_763_097_09 {
            return Ok(d);
        }
        current = difference(&current);
        crate::validation::validate_finite(&current, "differenced y")?;
    }
    Ok(max_d)
}

fn variance(y: &[f64]) -> f64 {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / y.len() as f64
}

pub(super) fn nsdiffs(y: &[f64], period: usize, max_d: usize) -> usize {
    if period < 2 || max_d == 0 || y.len() < 2 * period {
        return 0;
    }
    let scale = y.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let scaled: Vec<_> = y.iter().map(|v| v / scale).collect();
    let original = variance(&scaled);
    if original == 0.0 {
        return 0;
    }
    let differenced: Vec<_> = scaled[period..]
        .iter()
        .zip(&scaled)
        .map(|(a, b)| a - b)
        .collect();
    usize::from(variance(&differenced) / original < 0.64)
}

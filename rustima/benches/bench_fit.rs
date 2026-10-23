use criterion::{criterion_group, criterion_main, Criterion};
use rustima::optimizer;
use rustima::types::{SarimaxConfig, SarimaxOrder, Trend};

/// Deterministic LCG data generator (fixed seed → reproducible data).
fn generate_ar1_data(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let mut state: u64 = seed;
    for t in 1..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
        y[t] = phi * y[t - 1] + u;
    }
    y
}

fn bench_fit_ar1(c: &mut Criterion) {
    let data = generate_ar1_data(500, 0.65, 42);
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: true,
        enforce_invertibility: true,
        concentrate_scale: true,
        simple_differencing: false,
    };

    c.bench_function("fit_ar1_n500", |b| {
        b.iter(|| optimizer::fit(&data, &config, None, Some("lbfgs"), None, None).unwrap())
    });
}

fn bench_fit_arima111(c: &mut Criterion) {
    let data = generate_ar1_data(500, 0.3, 42);
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 1, 1, 0, 0, 0, 0),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: true,
        enforce_invertibility: true,
        concentrate_scale: true,
        simple_differencing: false,
    };

    c.bench_function("fit_arima111_n500", |b| {
        b.iter(|| optimizer::fit(&data, &config, None, Some("lbfgs"), None, None).unwrap())
    });
}

fn bench_fit_sarima_111_111_12(c: &mut Criterion) {
    // Cumulative sum of LCG noise to simulate integrated series
    let mut data = generate_ar1_data(500, 0.0, 42);
    for t in 1..data.len() {
        data[t] += data[t - 1];
    }
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 1, 1, 1, 1, 1, 12),
        n_exog: 0,
        trend: Trend::None,
        enforce_stationarity: false,
        enforce_invertibility: false,
        concentrate_scale: true,
        simple_differencing: false,
    };

    c.bench_function("fit_sarima111_111_12_n500", |b| {
        b.iter(|| optimizer::fit(&data, &config, None, Some("lbfgsb"), None, None).unwrap())
    });
}

fn bench_lbfgsb_comparison(c: &mut Criterion) {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/statsmodels_reference.json")).unwrap();
    let mut group = c.benchmark_group("lbfgsb_comparison");
    for name in ["ar1", "arma11", "arima111", "sarima_100_100_4"] {
        let case = &fixtures[name];
        let data: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let order: Vec<usize> = case["order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let seasonal: Vec<usize> = case["seasonal_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let config = SarimaxConfig {
            order: SarimaxOrder::new(
                order[0],
                order[1],
                order[2],
                seasonal[0],
                seasonal[1],
                seasonal[2],
                seasonal[3],
            ),
            concentrate_scale: true,
            enforce_stationarity: true,
            enforce_invertibility: true,
            ..SarimaxConfig::default()
        };
        let start: Vec<f64> = case["params"]
            .as_array()
            .unwrap()
            .iter()
            .take(order[0] + order[2] + seasonal[0] + seasonal[2])
            .map(|v| v.as_f64().unwrap())
            .collect();
        let methods = [
            "lbfgsb-strict",
            #[cfg(feature = "basin")]
            "basin-lbfgsb-strict",
        ];
        for method in methods {
            group.bench_function(criterion::BenchmarkId::new(name, method), |b| {
                b.iter(|| {
                    optimizer::fit(
                        std::hint::black_box(&data),
                        &config,
                        Some(&start),
                        Some(method),
                        Some(500),
                        None,
                    )
                    .unwrap()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fit_ar1,
    bench_fit_arima111,
    bench_fit_sarima_111_111_12,
    bench_lbfgsb_comparison
);
criterion_main!(benches);

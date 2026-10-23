use super::*;
use crate::test_helpers::load_fixtures;
use crate::types::SarimaxOrder;

fn fixture_data(case: &serde_json::Value) -> Vec<f64> {
    case["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn method_aliases_match_from_identical_starts() {
    let fixtures = load_fixtures();
    for name in ["ar1", "arma11", "arima111", "sarima_100_100_4"] {
        let case = &fixtures[name];
        let data = fixture_data(case);
        let order = case["order"].as_array().unwrap();
        let seasonal = case["seasonal_order"].as_array().unwrap();
        let mut config = SarimaxConfig {
            order: SarimaxOrder::new(
                order[0].as_u64().unwrap() as usize,
                order[1].as_u64().unwrap() as usize,
                order[2].as_u64().unwrap() as usize,
                seasonal[0].as_u64().unwrap() as usize,
                seasonal[1].as_u64().unwrap() as usize,
                seasonal[2].as_u64().unwrap() as usize,
                seasonal[3].as_u64().unwrap() as usize,
            ),
            ..SarimaxConfig::default()
        };
        for enforce in [false, true] {
            config.enforce_stationarity = enforce;
            config.enforce_invertibility = enforce;
            for concentrate in [false, true] {
                config.concentrate_scale = concentrate;
                let start = compute_start_params(&data, &config, None).unwrap();
                let legacy = fit(
                    &data,
                    &config,
                    Some(&start),
                    Some("lbfgsb-strict"),
                    Some(500),
                    None,
                )
                .unwrap();
                let basin = fit(
                    &data,
                    &config,
                    Some(&start),
                    Some("basin-lbfgsb-strict"),
                    Some(500),
                    None,
                )
                .unwrap();
                assert!(
                    basin.converged && legacy.converged,
                    "{name}, enforce={enforce}, concentrate={concentrate}"
                );
                assert!(
                    (basin.loglike - legacy.loglike).abs() <= 1e-4,
                    "{name}: Basin LL={}, legacy LL={}",
                    basin.loglike,
                    legacy.loglike
                );
                for (a, b) in basin.params.iter().zip(&legacy.params) {
                    assert!((a - b).abs() <= 1e-3 + 1e-3 * b.abs(), "{name}: {a} vs {b}");
                }
                assert_eq!(basin.method, "basin-lbfgsb-strict");
                assert!(basin.n_iter <= 500);
            }
        }
    }
}

#[test]
fn basin_strategies_respect_work_budgets() {
    let data = fixture_data(&load_fixtures()["arma11"]);
    let config = SarimaxConfig {
        order: SarimaxOrder::new(1, 0, 1, 0, 0, 0, 0),
        ..SarimaxConfig::default()
    };
    let start = [0.1, 0.1, 1.0];
    for method in [
        "basin-lbfgsb",
        "basin-lbfgsb-strict",
        "basin-lbfgsb-multi",
        "basin-lbfgsb-adaptive",
        "basin-lbfgsb-hybrid",
    ] {
        for budget in [0, 1, 5, 50] {
            let result = fit(
                &data,
                &config,
                Some(&start),
                Some(method),
                Some(budget),
                None,
            )
            .unwrap();
            assert!(result.loglike.is_finite());
            assert!(
                result.n_iter <= budget,
                "{method}: {} > {budget}",
                result.n_iter
            );
            if budget <= 1 {
                assert!(!result.converged, "{method}");
            }
            assert!(result.method.contains("basin-"));
        }
    }
}

#[test]
fn basin_initialization_and_fast_paths() {
    let data = fixture_data(&load_fixtures()["ar1"]);
    let mut config = SarimaxConfig {
        order: SarimaxOrder::new(1, 0, 0, 0, 1, 0, 40),
        ..SarimaxConfig::default()
    };
    let warm = run_sd_warm_start(&data, &config, None).unwrap();
    assert_eq!(warm.len(), 2);
    assert!(warm.iter().all(|x| x.is_finite()));
    config.order = SarimaxOrder::new(3, 0, 0, 0, 0, 0, 0);
    config.concentrate_scale = true;
    assert_eq!(
        fit(&data, &config, None, Some("basin-lbfgsb"), Some(500), None)
            .unwrap()
            .method,
        "burg-direct"
    );
    config.order = SarimaxOrder::new(0, 0, 0, 0, 0, 0, 0);
    let result = fit(&data, &config, None, Some("basin-lbfgsb"), Some(500), None).unwrap();
    assert!(result.converged);
    assert_eq!(result.n_iter, 0);
    assert_eq!(result.method, "basin-lbfgsb");
}

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

fn fixture_params(case: &serde_json::Value) -> Vec<f64> {
    case["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// Every accepted method name, in `Method::parse` table order (plain + basin- aliases).
const ALL_METHOD_NAMES: &[&str] = &[
    "lbfgsb",
    "lbfgsb-multi",
    "lbfgsb-adaptive",
    "lbfgsb-hybrid",
    "lbfgsb-strict",
    "lbfgsb_single",
    "lbfgs",
    "bfgs",
    "trust-region",
    "profile-trust-region",
    "nelder-mead",
    "nm",
    "basin-lbfgsb",
    "basin-lbfgsb-strict",
    "basin-lbfgsb-multi",
    "basin-lbfgsb-adaptive",
    "basin-lbfgsb-hybrid",
];

/// Local FP-identity tool for refactors: `cargo test -p rustima-core dump_characterization -- --ignored --nocapture`
/// before and after a change, then diff. Bit patterns differ across platforms, so nothing is asserted.
#[test]
#[ignore]
fn dump_characterization() {
    let fixtures = load_fixtures();
    for name in ["ar1", "arma11", "arima111", "sarima_100_100_4", "sarima_111_111_12"] {
        let case = &fixtures[name];
        let data = fixture_data(case);
        let order = case["order"].as_array().unwrap();
        let seasonal = case["seasonal_order"].as_array().unwrap();
        let config = SarimaxConfig {
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
        let out = Prepared::new(&data, None, &config)
            .unwrap()
            .loglike(&SarimaxParams::from_flat(&fixture_params(case), &config).unwrap())
            .unwrap();
        println!("loglike {name} {:016x} {:?}", out.loglike.to_bits(), out.loglike);
        if !matches!(name, "arma11" | "sarima_100_100_4") {
            continue;
        }
        let start = [0.1, 0.1, 1.0];
        for method in ALL_METHOD_NAMES {
            match fit(&data, &config, Some(&start), Some(method), Some(500), None) {
                Ok(r) => println!(
                    "fit {name} {method} {} {:016x} {:?} {} {}",
                    r.method, r.loglike.to_bits(), r.loglike, r.n_iter, r.converged
                ),
                Err(e) => println!("fit {name} {method} ERR {e}"),
            }
        }
    }
}

/// Pinned on 96d28d8 (macOS arm64) from `dump_characterization`; 1e-5 abs on the LL value because
/// the argmin location, not the optimum value, is what differs across BLAS/libm.
#[test]
fn every_method_name_is_characterized() {
    // (name, label, LL arma11, LL sarima_100_100_4) with start [0.1, 0.1, 1.0], maxiter 500.
    const EXPECTED: &[(&str, &str, f64, f64)] = &[
        ("lbfgsb", "basin-lbfgsb", -266.87450273204297, -285.3169115713603),
        ("lbfgsb-multi", "basin-lbfgsb-multi", -266.87450273005686, -285.31691157042223),
        ("lbfgsb-adaptive", "basin-lbfgsb-adaptive", -266.87450273007823, -285.3169115704433),
        ("lbfgsb-hybrid", "basin-lbfgsb-hybrid", -266.87450273005686, -285.3169115704218),
        ("lbfgsb-strict", "basin-lbfgsb-strict", -266.87450273204297, -285.3169115713603),
        ("lbfgsb_single", "basin-lbfgsb-strict", -266.87450273204297, -285.3169115713603),
        ("lbfgs", "lbfgs", -266.87450273004526, -285.3169115704214),
        ("bfgs", "bfgs", -266.8745027300454, -285.3169115704226),
        ("trust-region", "trust-region", -266.87450273005663, -285.31691157034754),
        // profile-trust-region degrades to plain trust-region when there is no exog
        ("profile-trust-region", "trust-region", -266.87450273005663, -285.31691157034754),
        ("nelder-mead", "nelder-mead", -266.87450363756795, -285.31691385888195),
        ("nm", "nelder-mead", -266.87450363756795, -285.31691385888195),
        ("basin-lbfgsb", "basin-lbfgsb", -266.87450273204297, -285.3169115713603),
        ("basin-lbfgsb-strict", "basin-lbfgsb-strict", -266.87450273204297, -285.3169115713603),
        ("basin-lbfgsb-multi", "basin-lbfgsb-multi", -266.87450273005686, -285.31691157042223),
        ("basin-lbfgsb-adaptive", "basin-lbfgsb-adaptive", -266.87450273007823, -285.3169115704433),
        ("basin-lbfgsb-hybrid", "basin-lbfgsb-hybrid", -266.87450273005686, -285.3169115704218),
    ];
    assert_eq!(EXPECTED.len(), ALL_METHOD_NAMES.len());
    let fixtures = load_fixtures();
    let arma11 = fixture_data(&fixtures["arma11"]);
    let sarima = fixture_data(&fixtures["sarima_100_100_4"]);
    let cases = [
        (&arma11, SarimaxOrder::new(1, 0, 1, 0, 0, 0, 0), 2),
        (&sarima, SarimaxOrder::new(1, 0, 0, 1, 0, 0, 4), 3),
    ];
    let start = [0.1, 0.1, 1.0];
    for (name, label, ll_a, ll_s) in EXPECTED {
        for (data, order, idx) in &cases {
            let config = SarimaxConfig { order: order.clone(), ..SarimaxConfig::default() };
            let r = fit(data, &config, Some(&start), Some(name), Some(500), None).unwrap();
            let want = if *idx == 2 { *ll_a } else { *ll_s };
            assert_eq!(r.method, *label, "{name}");
            assert!(r.converged, "{name} on {order:?}");
            assert!((r.loglike - want).abs() <= 1e-5, "{name} on {order:?}: {:?} vs {want:?}", r.loglike);
        }
    }
    for bad in ["basin-bfgs", "basin-lbfgsb_single", "basin-nm", "invalid", ""] {
        assert!(validate_method(bad).is_err(), "{bad:?} must be rejected");
    }
}

/// Port of the deleted `test_profile_trust_region.py::test_ptr_loglike_matches_pre_batched_baseline`.
/// Tolerances as in the original: LL 1e-8 abs, params 1e-4 rel (optimum sits on a flat ridge,
/// so the argmin moves ~5e-6 across BLAS/libm while the LL value does not).
#[test]
fn ptr_loglike_matches_deleted_python_baseline() {
    let case: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/ptr_baseline.json")).unwrap();
    let y: Vec<f64> = case["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let exog: Vec<Vec<f64>> = case["exog"]
        .as_array()
        .unwrap()
        .iter()
        .map(|col| col.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let config = SarimaxConfig {
        order: SarimaxOrder::new(2, 0, 0, 1, 1, 0, 24),
        n_exog: 2,
        enforce_stationarity: true,
        enforce_invertibility: true,
        ..SarimaxConfig::default()
    };
    let r = fit(&y, &config, None, Some("profile-trust-region"), Some(500), Some(&exog)).unwrap();
    let baseline_ll = case["baseline_loglike"].as_f64().unwrap();
    assert!(
        (r.loglike - baseline_ll).abs() <= 1e-8,
        "PTR LL drifted: got {:?}, baseline {:?}",
        r.loglike,
        baseline_ll
    );
    for (got, want) in r.params.iter().zip(case["baseline_params"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap())) {
        assert!((got - want).abs() <= 1e-4 * want.abs(), "param {got} vs baseline {want}");
    }
}

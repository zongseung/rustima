use super::*;

#[test]
fn adf_matches_statsmodels_reference() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/adf_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let y: Vec<_> = case["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let (stat, lag) = differencing::adf_statistic(&y).unwrap();
        assert!(
            (stat - case["stat"].as_f64().unwrap()).abs() < 1e-7,
            "ADF {stat}: {case}"
        );
        assert_eq!(lag, case["lag"].as_u64().unwrap() as usize);
        assert_eq!(
            differencing::ndiffs(&y, 2).unwrap(),
            case["d"].as_u64().unwrap() as usize
        );
    }
    assert_eq!(differencing::ndiffs(&[1.0; 20], 2).unwrap(), 0);
    assert_eq!(differencing::ndiffs(&[1.0, 2.0], 2).unwrap(), 0);
}

#[test]
fn seasonal_difference_and_disabled_detection() {
    let y: Vec<_> = (0..100)
        .map(|i| (i % 4) as f64 + (i / 4) as f64 * 0.01)
        .collect();
    assert_eq!(differencing::nsdiffs(&y, 4, 1), 1);
    assert_eq!(differencing::nsdiffs(&y, 4, 0), 0);
    assert_eq!(differencing::nsdiffs(&y, 0, 1), 0);
    assert_eq!(differencing::ndiffs(&y, 0).unwrap(), 0);
}

fn fake_fit(ic: f64, converged: bool) -> FitResult {
    FitResult {
        params: vec![],
        loglike: -ic / 2.0,
        scale: 1.0,
        n_obs: 100,
        n_params: 0,
        n_iter: 1,
        converged,
        method: "test".into(),
        aic: ic,
        bic: ic,
        warnings: vec![],
    }
}

#[test]
fn selection_preserves_soft_preference_and_failure_history() {
    for (nonconverged_ic, expected) in [(98.0, 100.0), (95.0, 95.0)] {
        let mut state = Selection::default();
        let order = SarimaxOrder::new(1, 0, 0, 0, 0, 0, 0);
        state.accept(order.clone(), Ok(fake_fit(100.0, true)), Criterion::Aic);
        state.accept(
            order.clone(),
            Ok(fake_fit(nonconverged_ic, false)),
            Criterion::Aic,
        );
        state.accept(
            order,
            Err(SarimaxError::InvalidInput("failed candidate".into())),
            Criterion::Aic,
        );
        let result = state.finish(0, 0);
        assert_eq!(result.best.unwrap().ic, expected);
        assert!(result.history[2].error.is_some());
        assert_eq!(result.history[2].ic, None);
    }
    let mut state = Selection::default();
    state.accept(
        SarimaxOrder::new(0, 0, 0, 0, 0, 0, 0),
        Ok(fake_fit(f64::NAN, true)),
        Criterion::Aic,
    );
    assert!(state.finish(0, 0).best.is_none());
}

#[test]
fn grid_matches_explicit_candidate_fits() {
    let fixtures = crate::test_helpers::load_fixtures();
    let y: Vec<_> = fixtures["ar1"]["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let config = SarimaxConfig {
        enforce_stationarity: true,
        enforce_invertibility: true,
        ..SarimaxConfig::default()
    };
    let options = SearchOptions {
        max_p: 1,
        max_q: 1,
        d: Some(0),
        dd: Some(0),
        stepwise: false,
        ..SearchOptions::default()
    };
    let result = search(&y, &config, &options, None, Some(100), None).unwrap();
    assert_eq!(result.history.len(), 4);
    for candidate in &result.history {
        let c = SarimaxConfig {
            order: candidate.order.clone(),
            ..config.clone()
        };
        let fit = optimizer::fit(&y, &c, None, None, Some(100), None).unwrap();
        assert!((candidate.ic.unwrap() - fit.aic).abs() < 1e-10);
    }
    assert!(result.best.is_some());
}

#[test]
fn invalid_bounds_and_inputs_fail_before_search() {
    let config = SarimaxConfig::default();
    let mut options = SearchOptions::default();
    assert!(search(&[], &config, &options, None, None, None).is_err());
    assert!(search(&[f64::NAN], &config, &options, None, None, None).is_err());
    options.max_p = usize::MAX;
    assert!(search(&[1.0; 20], &config, &options, None, None, None).is_err());
    options.max_p = 1;
    options.period = 1;
    assert!(search(&[1.0; 20], &config, &options, None, None, None).is_err());
    options.period = 0;
    options.dd = Some(1);
    assert!(search(&[1.0; 20], &config, &options, None, None, None).is_err());
}

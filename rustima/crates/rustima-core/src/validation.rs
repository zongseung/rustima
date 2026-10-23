use crate::error::{Result, SarimaxError};
use crate::types::SarimaxConfig;

pub fn validate_request(
    y: &[f64],
    config: &SarimaxConfig,
    exog: Option<&[Vec<f64>]>,
) -> Result<()> {
    let o = &config.order;
    validate_order(o.p, o.d, o.q, o.pp, o.dd, o.qq, o.s, config.n_exog)?;
    if y.is_empty() || y.iter().any(|x| !x.is_finite()) {
        return Err(SarimaxError::InvalidInput(
            "y must be nonempty and finite".into(),
        ));
    }
    let cols = exog.unwrap_or(&[]);
    if cols.len() != config.n_exog {
        return Err(SarimaxError::InvalidInput(
            "xreg column count does not match the model".into(),
        ));
    }
    for col in cols {
        if col.len() != y.len() || col.iter().any(|x| !x.is_finite()) {
            return Err(SarimaxError::InvalidInput(
                "xreg must be finite and have one row per observation".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_data_shapes_and_orders() {
        let mut c = SarimaxConfig::default();
        assert!(validate_request(&[1.0, 2.0], &c, None).is_ok());
        assert!(validate_request(&[], &c, None).is_err());
        assert!(validate_request(&[f64::NAN], &c, None).is_err());
        c.n_exog = 1;
        assert!(validate_request(&[1.0], &c, None).is_err());
        assert!(validate_request(&[1.0], &c, Some(&[vec![1.0, 2.0]])).is_err());
        assert!(validate_request(&[1.0], &c, Some(&[vec![f64::INFINITY]])).is_err());
        c.order.p = 21;
        assert!(validate_request(&[1.0], &c, Some(&[vec![2.0]])).is_err());
    }
}

const MAX_K_STATES: usize = 1024;

// Hard upper bounds on individual model order parameters to prevent
// OOM / compute-DoS from adversarial or erroneous inputs.
const MAX_P: usize = 20;
const MAX_Q: usize = 20;
const MAX_D: usize = 3;
const MAX_PP: usize = 4;
const MAX_QQ: usize = 4;
const MAX_DD: usize = 2;
const MAX_S: usize = 365;
const MAX_N_EXOG: usize = 100;

pub fn validate_order(
    p: usize,
    d: usize,
    q: usize,
    pp: usize,
    dd: usize,
    qq: usize,
    s: usize,
    n_exog: usize,
) -> Result<()> {
    // Individual order bounds (DoS prevention)
    if p > MAX_P {
        return Err(SarimaxError::InvalidInput(format!(
            "AR order p={} exceeds maximum {}",
            p, MAX_P
        )));
    }
    if q > MAX_Q {
        return Err(SarimaxError::InvalidInput(format!(
            "MA order q={} exceeds maximum {}",
            q, MAX_Q
        )));
    }
    if d > MAX_D {
        return Err(SarimaxError::InvalidInput(format!(
            "differencing order d={} exceeds maximum {}",
            d, MAX_D
        )));
    }
    if pp > MAX_PP {
        return Err(SarimaxError::InvalidInput(format!(
            "seasonal AR order P={} exceeds maximum {}",
            pp, MAX_PP
        )));
    }
    if qq > MAX_QQ {
        return Err(SarimaxError::InvalidInput(format!(
            "seasonal MA order Q={} exceeds maximum {}",
            qq, MAX_QQ
        )));
    }
    if dd > MAX_DD {
        return Err(SarimaxError::InvalidInput(format!(
            "seasonal differencing D={} exceeds maximum {}",
            dd, MAX_DD
        )));
    }
    if s > MAX_S {
        return Err(SarimaxError::InvalidInput(format!(
            "seasonal period s={} exceeds maximum {}",
            s, MAX_S
        )));
    }
    if n_exog > MAX_N_EXOG {
        return Err(SarimaxError::InvalidInput(format!(
            "n_exog={} exceeds maximum {}",
            n_exog, MAX_N_EXOG
        )));
    }
    if (pp > 0 || dd > 0 || qq > 0) && s < 2 {
        return Err(SarimaxError::InvalidInput(
            "seasonal order (P,D,Q) > 0 requires seasonal period s >= 2".into(),
        ));
    }

    // Overflow checking for state-space dimensions
    let k_ar = p
        .checked_add(s.checked_mul(pp).ok_or_else(|| {
            SarimaxError::InvalidInput("order overflow while computing k_ar".into())
        })?)
        .ok_or_else(|| SarimaxError::InvalidInput("order overflow while computing k_ar".into()))?;
    let k_ma = q
        .checked_add(s.checked_mul(qq).ok_or_else(|| {
            SarimaxError::InvalidInput("order overflow while computing k_ma".into())
        })?)
        .ok_or_else(|| SarimaxError::InvalidInput("order overflow while computing k_ma".into()))?;
    let k_order = std::cmp::max(
        k_ar,
        k_ma.checked_add(1).ok_or_else(|| {
            SarimaxError::InvalidInput("order overflow while computing k_order".into())
        })?,
    );
    let k_states_diff = d
        .checked_add(s.checked_mul(dd).ok_or_else(|| {
            SarimaxError::InvalidInput("order overflow while computing k_states_diff".into())
        })?)
        .ok_or_else(|| {
            SarimaxError::InvalidInput("order overflow while computing k_states_diff".into())
        })?;
    let k_states = k_order.checked_add(k_states_diff).ok_or_else(|| {
        SarimaxError::InvalidInput("order overflow while computing k_states".into())
    })?;
    if k_states > MAX_K_STATES {
        return Err(SarimaxError::InvalidInput(format!(
            "model too large: k_states={} exceeds limit {}",
            k_states, MAX_K_STATES
        )));
    }

    Ok(())
}

# rustima

[![CI](https://github.com/zongseung/rustima/actions/workflows/ci.yml/badge.svg)](https://github.com/zongseung/rustima/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![R](https://img.shields.io/badge/R-4.2%2B%20preview-blue)
![Rust](https://img.shields.io/badge/Rust-1.87%2B-orange)

[English](README.md) · [개발 안내](rustima/README.md) · [변경 이력](rustima/CHANGELOG.md)

Rust로 작성한 SARIMAX 엔진입니다. Python과 R에서 같은 코어의 상태공간
Kalman 우도·최적화·예측을 호출하며, 외생변수와 Rayon 기반 독립 작업 병렬화를
지원합니다. 기본 우도를 Whittle 근사로 대체하지 않습니다.

Python 정식 버전은 `0.4.0`이며, R `0.1.1`은 preview입니다.
L-BFGS-B(기본)·L-BFGS·BFGS·Nelder-Mead는 **Basin 3.0**을 사용합니다.
기존 C 최적화 코드와 빌드 스크립트를
제거했고, `lbfgsb`와 `basin-lbfgsb` 이름을 모두 유지했습니다.
trust-region·profile-trust-region도 공통 Rust 코어에 남아 있습니다.
L-BFGS-B의 `n_iter`는 line search를 포함한 목적함수·gradient 평가 횟수이며,
나머지 Basin 방법은 완료한 solver 반복 횟수를 셉니다.

## Python

공개된 Python 버전은 `pip install rustima`로 설치할 수 있습니다.
공개 버전과 이 개발 브랜치의 기능은 다를 수 있습니다. 체크아웃을 빌드하려면:

```sh
cd rustima
python -m pip install maturin
# 가상환경을 활성화한 뒤 실행합니다.
maturin develop --release --manifest-path packages/python/Cargo.toml
```

```python
import numpy as np
from rustima import SARIMAXModel, auto_arima

y = np.random.default_rng(10).normal(size=200)
fit = SARIMAXModel(y, order=(1, 0, 1)).fit()
print(fit.params, fit.aic)
print(fit.forecast(5))
selected = auto_arima(y, seasonal=False)
print(selected.order)
```

Python은 외생변수, 추론·진단, rolling 예측, batch 적합 및 stepwise/grid
Auto ARIMA를 제공합니다. 자세한 빌드·검증 절차는 [개발 안내](rustima/README.md)를 참고하세요.

## R preview

R 패키지는 배포 준비 단계이며 아직 CRAN에 게시하지 않았습니다.
R >= 4.2, Rust >= 1.87, Cargo와 운영체제에 맞는 R 빌드 도구가 필요합니다.
Windows에서는 Rtools가 필요합니다.

```sh
cd rustima
python3 tools/build_r_package.py
R CMD INSTALL dist/r/rustima_0.1.1.tar.gz
```

빌더가 공통 코어와 잠긴 Rust 의존성을 소스 패키지에 묶습니다.
설치 중 Cargo는 offline 모드로 컴파일합니다. Python은 소스 패키지 생성에만
필요하고, R의 실행 경로는 extendr를 통해 Rust를 직접 호출합니다.

```r
library(rustima)
set.seed(10)
y <- arima.sim(list(ar = 0.5), n = 120)
fit <- sarimax(y, c(1, 0, 0))
coef(fit)
AIC(fit)
predict(fit, n.ahead = 5)$pred
fits <- sarimax_batch(list(y, y), c(1, 0, 0), nthreads = 2)
grid <- sarimax_grid_search(y, list(c(1, 0, 0), c(2, 0, 0)), nthreads = 2)
rustima_backend_info()
```

초기 R API는 고정 차수 적합·우도·예측·batch·명시적 후보 grid search를 제공합니다.
자동 차분 선택과 stepwise Auto ARIMA는 현재 Python에서 제공합니다.
적합 후 `converged`와 `warnings`를 확인하세요. 결측값과 비유한 관측값은 거부합니다.

## 디렉터리와 태그

| 경로 | 역할 |
|---|---|
| `rustima/crates/rustima-core/` | Python/R 공통 통계 엔진 |
| `rustima/packages/python/` | Python 바인딩·패키지·테스트 |
| `rustima/packages/r/rustima/` | R 바인딩·패키지·문서·테스트 |
| `rustima/tools/build_r_package.py` | 독립 설치 가능한 R 소스 패키지 생성 |

릴리스 태그는 Python `py-v…`, R `r-v…`로 분리하고 후보 버전은 `-rc.N`을 붙입니다.
R 태그는 PyPI 게시 경로를 실행하지 않습니다. R 릴리스 자동화는 소스 빌드와
검증 결과를 만들며, CRAN 제출은 별도 단계입니다.
[개발 안내](rustima/README.md)를 참고하세요.

[GPL-3.0-or-later](rustima/LICENSE) 라이선스를 사용합니다.

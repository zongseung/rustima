# Submission comments

This is the first CRAN submission of rustima.

The package fits seasonal ARIMA models with optional regression terms using
a shared Rust engine. It provides likelihood evaluation, forecasting,
independent batch fits, and automatic order selection without a Python runtime.

## Test environments

Full `R CMD check --as-cran` was run on the same source archive in all of
these environments on 2026-10-26:

- Ubuntu 24.04.5, x86_64: R 4.6.1.
- Ubuntu 24.04.5, x86_64: R-devel (2026-10-24 r90762).
- macOS Tahoe 26.6.2, arm64: R 4.6.1.
- Windows Server 2022, x86_64: R 4.6.1 UCRT.
- Windows Server 2022, x86_64: R-devel (2026-10-24 r90762 UCRT).
- Local macOS arm64: R 4.5.1 with the minimum Cargo/rustc 1.87.0.

Every check completed with 0 errors, 0 warnings, and 1 NOTE. Both R test
scripts, compiled Rust checks, and PDF and HTML manuals passed.
The source archive was built using R 4.6.1. Windows R-devel was tested
in GitHub Actions. The same archive was additionally uploaded to Win-builder
R-devel on 2026-10-26; its result email is pending.

Check results:
https://github.com/zongseung/rustima/actions/runs/40275011224

## Notes for reviewers

- The sole NOTE is `New submission`, with the maintainer information.
- The source archive is 5,197,682 bytes. Most of its size is the bundled
  Rust dependencies in `src/rust/vendor.tar.xz`, enabling offline builds.
  Installation downloads no Rust crates or precompiled binaries.
- Cargo and rustc >= 1.87 are checked during configuration. Compilation
  uses one worker; examples and tests use at most two Rayon workers.
- Bundled Rust authorship and licensing are indexed in `inst/COPYRIGHTS`
  and referenced by DESCRIPTION. Original notices are retained, with
  extendr's matching upstream license and contributor list additionally
  supplied in `inst/extendr`.
- Case-insensitive checks found no package-name conflict in CRAN's current
  and archived packages or Bioconductor's release and devel packages.

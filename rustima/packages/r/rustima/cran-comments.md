# Submission comments — DRAFT, update after final platform checks

This is the first CRAN submission of rustima.

The package fits seasonal ARIMA models with optional regression terms using
a shared Rust engine. It provides likelihood evaluation, forecasting,
independent batch fits, and automatic order selection without a Python runtime.

## Test environments

- Local macOS arm64: R 4.5.1, Cargo/rustc 1.87.0, TinyTeX (TeX Live 2026),
  HTML Tidy 5.8.0. Full `R CMD check --as-cran`: 0 errors, 0 warnings, 1 NOTE.
  Offline installation, both R test scripts, PDF and HTML manuals all pass.
- R-release on Linux, macOS, and Windows: final source revision not yet checked.
- R-devel on Linux and Win-builder: not yet checked.

Replace the above with results for the final source archive before uploading.
Local R 4.5.1 is older than the current release and does not replace R-devel
checks. The R source release workflow includes the full PDF manual check.

## Notes for reviewers

- A `New submission` NOTE is expected for this first submission.
- The checked source archive is 5,199,895 bytes. The incoming NOTE reports
  the maintainer, new submission, and source archive size.
- Rust dependencies are bundled as `src/rust/vendor.tar.xz` for offline builds,
  which accounts for most of the source archive size. Installation does not
  download Rust crates or precompiled binaries.
- Both Cargo and rustc >= 1.87 are checked during configuration. Compilation
  uses one worker; examples and tests use at most two Rayon workers.
- Bundled Rust authorship and licensing are recorded in `inst/COPYRIGHTS` and
  referenced by DESCRIPTION. Original notices are retained, with extendr's
  license and contributor list additionally supplied in `inst/extendr`.

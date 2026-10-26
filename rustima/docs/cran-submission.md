# CRAN submission preparation: rustima 0.1.1

This is preparation for the first CRAN submission, not a record of submission
or acceptance. The existing GitHub R preview is a separate distribution.
The maintainer confirmed `zongseung <new9279@gachon.ac.kr>` on 2026-10-26.

## Build the submission archive

Run from the `rustima/` workspace with R >= 4.2, Cargo/rustc >= 1.87,
Python 3, and the platform's R build tools. The builder needs access to the
locked Cargo sources; the resulting package installs without network access.

```sh
python3 tools/build_r_package.py --output dist/r/cran
mkdir -p dist/r/cran/check
RAYON_NUM_THREADS=2 R CMD check --as-cran \
  --output=dist/r/cran/check dist/r/cran/rustima_0.1.1.tar.gz
```

Use a current R release to build the archive and R-devel to check it before
submission. A working LaTeX installation is required for the complete check;
`--no-manual` does not verify the PDF manual. Keep the checked archive,
`00check.log`, `00install.out`, test logs, and PDF together. Record its SHA-256:

```sh
shasum -a 256 dist/r/cran/rustima_0.1.1.tar.gz
```

The package compiles its bundled Rust sources with `--locked --offline -j 1`.
No compiler, Rust crate, or precompiled library is downloaded during installation.
Rust crate credits are indexed in `inst/COPYRIGHTS`; notices remain in the
vendored sources. The extendr 0.9.0 crates omit their repository's license files,
so the upstream license and contributor list are also supplied in `inst/extendr`.
Their source commit is recorded in the generated `inst/COPYRIGHTS`.

## Local validation, 2026-10-26

The final archive was checked on macOS arm64 with R 4.5.1 and the minimum
supported Cargo/rustc 1.87.0, using TinyTeX (TeX Live 2026) and HTML Tidy 5.8.0.
`R CMD check --as-cran` completed with **0 errors, 0 warnings, 1 NOTE**.
The NOTE reports the maintainer, new submission, and source archive size.
Offline installation, `auto.R`, `smoke.R`, and the PDF and HTML manuals pass.

The source archive is `dist/r/cran/rustima_0.1.1.tar.gz` (5,199,895 bytes).
Its SHA-256 is:

```text
b23d00ef47c975844104fa8257bdaf14021a0c8a1749a942ac92aa95dfbb29d7
```

Logs are under `dist/r/cran/check-macos-final/rustima.Rcheck/`; a separate
manual is saved as `dist/r/cran/rustima-manual.pdf`. The archive was also
checked for compiled binaries, retained license notices for its 86 vendored
crates, exclusion of `cran-comments.md`, and agreement with the package sources.
Configuration self-checks reject old Cargo and old rustc separately and accept
the declared minimum versions. `actionlint` passes for the release workflow.

This local R version is older than the current release. Final current-R,
R-devel, Linux, Windows, and Win-builder checks are still required. Earlier
GitHub CI passed on all three operating systems, but predates these changes
and used `--no-manual`; it is not a check of this final archive.

## Checks before upload

- The R release workflow checks R-release on Linux, macOS, and Windows, plus
  R-devel on Linux. It includes the PDF manual and rejects errors and warnings.
  Its artifacts include the source archive and check logs.
- Run this workflow on the final source revision, then review every NOTE.
  Results from an earlier revision do not validate the final archive.
- Check the exact final archive on
  [Win-builder](https://win-builder.r-project.org/) with R-devel. The service
  sends results to the maintainer; this has not been requested during local
  preparation.
- Update `packages/r/rustima/cran-comments.md` with those results. It is kept
  outside the source archive by `.Rbuildignore`.
- Confirm that no submission of this package is already pending. First-submission
  notes and the size needed to bundle Rust sources should be explained in the
  submission comments. Do not suppress checks to remove notes.
- Recheck name availability in CRAN (including the archive) and Bioconductor.
  The local preparation found no match in CRAN's current package index and the
  `rustima` archive URL returned HTTP 404. Bioconductor's package-index download
  returned HTTP 403 and its complete name check remains pending.

GitHub Actions can be run from the R source release workflow's **Run workflow**
button after these changes are committed and pushed. A tag is not needed for
the check. This workflow does not upload to CRAN.

## Submit after final checks

Upload the checked `rustima_0.1.1.tar.gz` using the
[CRAN submission form](https://cran.r-project.org/submit.html), with the final
contents of `cran-comments.md` in the optional comment. The designated
maintainer must accept CRAN's confirmation email. Wait for a response before
submitting another revision; increase the version if a revised submission is
needed. Do not describe the package as CRAN-published until acceptance.

References: [CRAN policy](https://cran.r-project.org/web/packages/policies.html),
[submission checklist](https://cran.r-project.org/web/packages/submission_checklist.html),
and [Rust requirements](https://cran.r-project.org/web/packages/using_rust.html).

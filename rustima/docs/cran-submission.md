# CRAN submission: rustima 0.1.1

Final validation completed on 2026-10-26. The maintainer confirmed
`zongseung <new9279@gachon.ac.kr>`. A CRAN submission request was sent on
2026-10-26. The maintainer completed email confirmation and reported CRAN's
message that the package was forwarded to the CRAN submission team.

## Checked submission archive

Use `dist/r/cran/final/rustima_0.1.1.tar.gz` (5,197,682 bytes).
This is the single archive built with R 4.6.1 and checked in every environment
below. Earlier archives under `dist/r/cran/` are historical candidates.

SHA-256:

```text
5af9b052597f6d3b797cc8ca61e86ef13b182161fad3a950973b20d403cf5534
```

Source commit: `0e9e2342c00649d9c26f283bd35c56939b0f6867`.
[Successful GitHub Actions run](https://github.com/zongseung/rustima/actions/runs/40275011224).
The workflow builds one archive and verifies its SHA-256 in each check job.

## Final check results

All six full `R CMD check --as-cran` runs finished with **0 errors,
0 warnings, 1 NOTE**. The sole NOTE is the incoming new-submission note.
Both R test scripts (`auto.R`, `smoke.R`), compiled Rust checks, and PDF
and HTML manuals pass in every environment.

| Environment | R |
|---|---|
| Ubuntu 24.04.5, x86_64 | 4.6.1 |
| Ubuntu 24.04.5, x86_64 | R-devel, 2026-10-24 r90762 |
| macOS Tahoe 26.6.2, arm64 | 4.6.1 |
| Windows Server 2022, x86_64 | 4.6.1 UCRT |
| Windows Server 2022, x86_64 | R-devel, 2026-10-24 r90762 UCRT |
| Local macOS arm64, minimum Rust toolchain | 4.5.1; Cargo/rustc 1.87.0 |

Downloaded platform logs and manuals are under
`dist/r/cran/final/platform-checks/`; local minimum-toolchain logs, including
`00install.out`, are under `dist/r/cran/final/check-macos-msrv/rustima.Rcheck/`.
A machine-readable summary is `dist/r/cran/final/check-summary.json`.

Archive checks confirm the SHA-256, preserved extendr notices, absence of
compiled binaries, exclusion of `cran-comments.md`, and absence of Codex/OpenAI
authorship. `actionlint` passes. The source workflow now also retains
`00install.out` in future runs; that artifact-retention change does not alter
the checked package sources or archive.

## Packaging and name checks

The builder embeds the shared Rust core and 86 locked Rust crates. Installation
uses a package-local Cargo home and `--locked --offline -j 1`; no Rust compiler,
crate, or precompiled library is downloaded. Configuration checks Cargo and
rustc >= 1.87. Credits are indexed in `inst/COPYRIGHTS`; original licenses remain
in the vendored sources. Extendr 0.9.0 omits its repository license files from
its crates, so the matching upstream license and contributor list are also
supplied in `inst/extendr`, with their source commit recorded in the index.

Case-insensitive name checks on 2026-10-26 found no `rustima` match in CRAN's
25,559 current packages or 28,325 archive names, or Bioconductor's release
(3,818 package links) and devel (3,874 package links). Evidence is saved in
`dist/r/cran/name-checks.json`.

## Submission status, 2026-10-26

- [PR #12](https://github.com/zongseung/rustima/pull/12) was merged into `main`
  as `9e8407a2a526bc4e5eda7daa0dd2376652ffb015`.
  [The main-branch CI passed](https://github.com/zongseung/rustima/actions/runs/40278546172).
- The same checked archive was uploaded to the separate Win-builder R-devel
  service at 06:14 UTC. Its upload response reports the correct filename and
  5,197,682 bytes. Results will be sent to `new9279@gachon.ac.kr`.
- CRAN accepted the submission request at 06:24 UTC and displayed Step 3
  (Confirmation). Its response says a confirmation email was sent to the
  maintainer. The maintainer then completed the email confirmation and supplied
  the success message: the package was uploaded to the CRAN submission team.
  The archive appeared in CRAN's incoming/pretest queue; downloading it confirmed
  that its SHA-256 exactly matches the checked candidate above.
  **Submission is confirmed and awaiting CRAN review; publication is pending.**

Receipts, the exact submitted comments, and a combined status record are saved
under `dist/r/cran/final/submissions/`. `cran-confirmation-response.html` records
CRAN's response; `win-builder-r-devel-response.html` records its upload receipt.
Do not upload another CRAN revision while this request is being processed.
Await the CRAN team's response.

All commits included in PR #12 use the existing author identity; no Codex or
OpenAI co-author trailer was added. Documentation and artifact-retention edits
after the tested commit do not change the package source in the checked archive.

References: [CRAN policy](https://cran.r-project.org/web/packages/policies.html),
[submission checklist](https://cran.r-project.org/web/packages/submission_checklist.html),
and [Rust requirements](https://cran.r-project.org/web/packages/using_rust.html).

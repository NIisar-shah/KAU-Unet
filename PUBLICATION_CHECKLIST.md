# Public-release checklist

## Completed in the repository

- [x] Apache License 2.0 added.
- [x] CAMUS data excluded through `.gitignore`; users are directed to the
  original provider.
- [x] Common checkpoint, run-output, credential, environment, and cache files
  excluded through `.gitignore`.
- [x] README documents installation, data layout, smoke testing, training,
  evaluation, five-seed aggregation, and clustered statistics.
- [x] Revised five-model analysis uses U-Net, KAN-UNet without tanh,
  MLP-bottleneck U-Net, KAN-UNet, and KAU-Net.
- [x] CPU smoke tests and GitHub Actions workflow added.
- [x] Provisional `CITATION.cff` metadata added.

## Author confirmation required before public release

- [ ] Confirm that all coauthors and the institution authorize code publication.
- [ ] Review the complete Git history for passwords, API keys, private data,
  identifiable data, proprietary code, or confidential assets.
- [ ] Test installation in a genuinely fresh Python environment.
- [ ] Run `python -m pytest -q tests` successfully.
- [ ] Verify one complete CAMUS training/evaluation run using the documented
  directory layout.
- [ ] Compare deposited numerical outputs against every manuscript table and
  figure.
- [ ] Complete `ENVIRONMENT.md` from the actual experiment machine.
- [ ] Decide whether checkpoints will be released. If not, remove any manuscript
  promise that checkpoints will be shared.
- [ ] Deposit non-identifiable result CSV files described in
  `results/README.md`.
- [ ] Confirm the manuscript uses the public repository URL in its
  Statements and Declarations section.
- [ ] Create signed or annotated release `v1.0.0`.
- [ ] Archive that release with Zenodo if a permanent DOI is desired.
- [ ] Add `date-released`, the archival DOI, and the final paper citation to
  `CITATION.cff` only after those values exist.

Unchecked items are deliberately not represented as completed.

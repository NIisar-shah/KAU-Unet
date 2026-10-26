# Public-release checklist (complete before clicking Publish)

- [ ] Select an open-source license **you are authorized to grant** and add `LICENSE`.
  If you leave the repository without a license, viewing is permitted under
  GitHub's terms but third-party reuse rights are not granted automatically.
- [ ] Verify that all coauthors and your institution permit code publication.
- [ ] Verify no passwords, API keys, private user data, identifiable data,
  unpublished proprietary code, or institution-owned confidential assets.
- [ ] Do **not** publish the CAMUS scans or masks; direct users to the source.
- [ ] Test installation in a fresh environment and inspect a CAMUS input example.
- [ ] Re-run at least a model forward pass; ideally reproduce a complete
  held-out evaluation on the original split and compare with the manuscript.
- [ ] Decide whether to publish trained checkpoints; these were NOT supplied.
- [ ] Update README with the paper citation, archival DOI, and tested versions.
- [ ] Replace any leftover institutional absolute paths in optional research scripts.
- [ ] Save a dated release and archive it through Zenodo when ready.
- [ ] Confirm the journal's policies regarding preprints and public code.

This checklist is intentionally not a declaration of completed reproducibility.

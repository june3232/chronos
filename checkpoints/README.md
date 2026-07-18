# Released checkpoints

This directory contains the validation-selected MCANN checkpoint for each
market in the fixed benchmark protocol:

- `be_best.pt`
- `de_best.pt`
- `fr_best.pt`
- `np_best.pt`
- `pjm_best.pt`

The files are stored with Git LFS. After cloning the repository, run
`git lfs pull` to download the checkpoint contents.

Each file is a PyTorch checkpoint containing the market name, selected epoch,
validation MAE, model settings, model parameters, and the complete model state
dictionary. SHA-256 digests are provided in `SHA256SUMS`.

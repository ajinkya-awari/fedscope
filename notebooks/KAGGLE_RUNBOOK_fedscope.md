# Project 05 FedScope Kaggle Runbook

This runbook accompanies `kaggle_run_fedscope.ipynb`. It is a manual, cell-by-cell execution
guide. The notebook is upload-ready but has not been executed on the laptop or in Kaggle.

## Project routing

Attach a Kaggle Dataset containing the complete Project 05 source bundle. The notebook does not
invent a Kaggle slug: Cell 2 searches `/kaggle/input` for the actual tree containing both
`kaggle/run_fedscope.py` and `kaggle/requirements.txt`. The working copy is
`/kaggle/working/fedscope_project`.

PathMNIST is not expected as an attached input dataset. The bounded runner requests PathMNIST size
28 through the project loader and stores its downloaded data under `/kaggle/working/medmnist`.
Runner outputs belong under `/kaggle/working/fedscope_artifacts`. The source layout is
`fedscope/`, `study/`, `tests/`, `kaggle/`, `notebooks/`, and control Markdown files.

The exact dependency file is `/kaggle/working/fedscope_project/kaggle/requirements.txt`:
`flwr[simulation]==1.20.0`, `medmnist==3.0.2`, and the bounded matplotlib range.

## Cell order and gates

Run cells manually in order. Each cell prints its status and next action. A `STOP` message means
do not skip ahead.

| Cell | Purpose | Safe without approval? |
| --- | --- | --- |
| 1 | Configuration, stage, blockers | Yes |
| 2 | Inspect actual Kaggle inputs and detect source tree | Yes |
| 3 | Preserve any prior working copy and copy source | Yes; it moves an existing working copy to a timestamped backup |
| 4 | Validate files and directories | Yes |
| 5 | Install the exact project requirements | Kaggle-only approved scope |
| 6 | Explain possible kernel restart | Yes; never restarts automatically |
| 7 | Remove live-provider variables and set local synthetic mode | Yes |
| 8 | Run synthetic validation | Yes after Cell 5 succeeds |
| 9 | Inspect only sanitized synthetic JSON | Yes |
| 10 | State the live-access boundary | Yes |
| 11 | Optional Kaggle Secret loading | Separate approval required; value is never printed |
| 12 | Bounded environment preflight | Explicit approval comment required |
| 13 | One-round Flower smoke | Explicit approval comment required |

The synthetic command is the project-documented offline suite:

```text
python -B -m pytest -q -p no:cacheprovider /kaggle/working/fedscope_project/tests
```

Cell 8 writes sanitized metadata to
`/kaggle/working/fedscope_project/evidence/synthetic/synthetic_validation_<UTC>.json`.
Cell 9 reads only the newest JSON and never prints test output or exception text.

The smallest live preflight is Cell 12. After the user changes its explicit approval comment,
it records Python/package versions, CUDA availability, and runner presence without loading
PathMNIST or starting training. Its sanitized evidence path is
`/kaggle/working/fedscope_project/evidence/preflight/preflight_<UTC>.json`.

The next approved project command is Cell 13:

```text
python /kaggle/working/fedscope_project/kaggle/run_fedscope.py --mode smoke --data-root /kaggle/working/medmnist --output-root /kaggle/working/fedscope_artifacts --device auto
```

This is a bounded federated training smoke: one round, seed 42, five clients. It is not the
three-seed benchmark. Cell 13 has a 30-minute timeout, writes only sanitized command metadata,
and never starts until its exact approval comment is supplied. Its output directories must be
inspected and retained.

## Kernel restart

After Cell 5, restart the Kaggle kernel once if pip changed a package that was already imported,
especially torch, Flower, MedMNIST, or matplotlib. Do not restart automatically. After restarting,
run Cell 7, then continue with Cell 8. Do not rerun Cell 3 unless the copied working tree was
intentionally preserved elsewhere.

## Live and release boundaries

The current project state is local implementation plus historical offline evidence. Flower/Kaggle
execution, real sample counts, partition manifests from PathMNIST, centralized metrics, federated
metrics, CUDA results, checkpoints, W&B, Hugging Face, deployment, publication, and release are
pending. No result may be claimed from this notebook without dated logs, sanitized evidence paths,
exit status, device, versions, seeds, sample counts, manifest hashes, and gate artifacts.

Do not run `--mode approved-seeds` until the smoke command and centralized gate pass and the user
explicitly approves the next gate. Never start overnight work automatically. Do not attach raw
patient data, use `.env` files, paste credentials, print provider responses, or delete evidence.

## Failure preservation

If a cell fails, stop at that cell. Keep the displayed evidence path, notebook output, and any
timestamped backup under `/kaggle/working`. Do not delete or overwrite prior evidence. Inspect only
sanitized JSON. Record the exit status and failure category in the project handover after the
Kaggle session. A failed synthetic validation blocks preflight; a failed preflight blocks the
smoke; a failed smoke blocks the three-seed run.

## Exact next action

Upload this notebook and the source bundle as a Kaggle Dataset, open the notebook in the
authenticated Kaggle session, and execute Cell 1. Then continue one cell at a time. The first
cell that requires a new explicit approval is Cell 12; Cell 13 requires a separate smoke approval.

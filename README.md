# Local Noise Optima and Size-Uniform Broadband Reservoir Costs

Research data and Python calculation code for the work by Chengxi Li, Hao Zhu,
and Wanzi Sun. Release **v1.1.0** accompanies the four-figure manuscript and
its finite-auxiliary numerical comparison. The earlier `v1.0.0` tag is retained.

A finite boundary region provides a noise-rate lower bound, while an explicit
reservoir extension provides a size-uniform achievable ceiling. Their separation
yields necessary rates and two noise targets whose minimum matrix range is
exactly two.

The single-frequency noise floor and unrestricted saturating reservoirs are
established prior results. This work adds a conditional adjacent-cell construction
and quantitative shared-band resource results for the two specified Jacobi drift
families. See [the scope and data dictionary](docs/DATA_DICTIONARY.md).

## Quick Reproduction

Use Python 3.11 or newer. The reference calculations used Python 3.13.13;
the dependency versions used for this release are pinned in `requirements.txt`.

```bash
git clone --branch v1.1.0 https://github.com/phycxli/Local-Noise-Optima-and-Size-Uniform-Broadband-Reservoir-Costs.git
cd Local-Noise-Optima-and-Size-Uniform-Broadband-Reservoir-Costs
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install the dependencies, verify the release, and regenerate the four main
figures and Fig. S1:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py manifest
python scripts/reproduce.py verify
python scripts/reproduce.py figures
python scripts/reproduce.py experiment
```

`verify` checks the saved matrices using exact rational arithmetic, rechecks
S5/S16/S17 on independent finite examples, checks the all-length witnesses and
extensions, and verifies the supplementary boundary-certificate artifacts.
It does not rerun the optimizer. The final line is `PASS: verify`; the report is
`reproduction/verification_report.json` and individual logs are in
`reproduction/logs/`. Rational checks can take several minutes.

`figures` regenerates `fig1_local_device`, `fig2_pointwise_noise`,
`fig3_shared_band_cost`, `fig4_design_targets`, and
`finite_frequency_local_optimality` as PDF and PNG under `reproduction/figures/`.
Fig. S1 uses archived CSVs without rerunning the pointwise optimizer.

`experiment` computes the specified finite-auxiliary, intrinsic-loss, and
thermal-bath comparison. Its report is
`reproduction/results/reports/experimental_realization_audit_2026_10_05.json`.
These are numerical predictions for a proposed device; no experimental data
were collected. Sampled gain minima are distinct from whole-band certificates.
All workflows run on a working copy under `reproduction/`, preserving the
published inputs and their recorded hashes.

## Organization

| Location | Contents |
|---|---|
| `src/` | Shared physical models, exact band Gram matrices, and tail bounds. |
| `scripts/` | Calculation, certification, and plotting programs; original names preserve certificate provenance. |
| `results/` | CSV data, parameter records, JSON certificates, and saved NPZ reservoir and certificate-factor arrays. |
| `figures/` | The published main and supplementary figure outputs. |
| `docs/REPRODUCIBILITY.md` | Commands, computational stages, outputs, and verification limits. |
| `docs/DATA_DICTIONARY.md` | Model parameters, observables, array meanings, and scope. |
| `docs/file_catalog.json` | File-level classification of the four scientific workflows. |
| `docs/VERIFICATION.json` | Saved-certificate, figure, and finite-auxiliary runs in the reference environment. |
| `release_manifest.json` | SHA256 and byte length of each release file. |
| `DATA_AVAILABILITY.md` | English and Chinese data and code availability statements. |

The four workflows are pointwise constructions and rate bounds, shared-band
optimization and size-uniform resources, supplementary physical response, and
supplementary boundary certificates. The full entry-point list is in
`docs/file_catalog.json`.

## Recompute Optimized Reservoirs

```bash
python scripts/reproduce.py optimize --output-dir reproduction-optimization
```

This longer workflow reruns the 27 finite-chain optimization problems, the 18
size-uniform prefix problems, and the two range-two prefix constructions, then
rechecks their certificates and plots the results. Solver outputs can differ
between platforms. Certification, rather than identical optimizer arrays, is
the acceptance criterion. See [reproduction details](docs/REPRODUCIBILITY.md).

## Citation and License

Use the `CITATION.cff` metadata and specify release `v1.1.0` when referring to
these data and code. The repository retains its existing [MIT license](LICENSE).
The fixed release can be downloaded from the tag's **Code / Download ZIP** menu.

## 中文说明

本仓库公开论文对应的计算代码、数值数据、储库矩阵和认证因子。四类工作流及
文件清单见 `docs/file_catalog.json`。先安装 `requirements.txt`，再运行
`python scripts/reproduce.py verify` 核验证书，运行
`python scripts/reproduce.py figures` 重绘四张正文图和图 S1，运行
`python scripts/reproduce.py experiment` 复现有限辅助器数值预测。计算输出集中写入
`reproduction/`，保留发布数据及其校验值。重新求解优化问题使用 `optimize`
工作流；模型适用条件、数据含义和补充材料复现命令见 `docs/`。

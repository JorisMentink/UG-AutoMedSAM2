# UG-AutoMedSAM2

**Uncertainty-guided, self-prompted MedSAM2 for prostate contour refinement.**

UG-AutoMedSAM2 explores whether segmentation uncertainty can guide automatic prompting of [MedSAM2](https://github.com/bowang-lab/MedSAM2) to refine existing prostate clinical target volume (CTV) contours. The intended application is to support clinicians and reduce manual editing burden in MR-Linac adapt-to-shape workflows.

This repository contains the inference and experimentation code developed for Joris Mentink's MSc Medical Engineering thesis at Eindhoven University of Technology. It uses existing nnU-Net segmentations and uncertainty maps as inputs; it does not train nnU-Net or generate its ensemble uncertainty maps.

> Research implementation. The code is provided for experimentation and further development, not clinical use.

## Method overview

The pipeline processes an MRI volume as a sequence of axial slices:

1. **Load inputs:** MRI, an initial segmentation, and its uncertainty map.
2. **Construct an uncertainty envelope:** threshold the uncertainty map using a target envelope thickness or a fixed threshold.
3. **Generate prompts:** use the coarse mask, uncertainty-guided positive/negative point pairs ("staples"), or bounding boxes.
4. **Run MedSAM2:** propagate predictions through the volume for each configured prompt strategy.
5. **Fuse predictions:** combine strategy logits using equal or custom weights and threshold the result.
6. **Evaluate:** compare predictions with a reference segmentation or a consensus derived from observer recontours.

The implementation supports binary and Gaussianized mask prompts, multiple propagation strategies, and restriction of propagation to the slice range containing the initial segmentation.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| [`LUNDPROBE_inference.ipynb`](LUNDPROBE_inference.ipynb) | Step-by-step inference and evaluation notebook. |
| [`DataLoader.py`](DataLoader.py) | Load LUND-PROBE images, masks, uncertainty maps, and observer recontours. |
| [`unc_handling.py`](unc_handling.py) | Construct uncertainty envelopes and generate prompts through `UG_prompter`. |
| [`uncertainty_util.py`](uncertainty_util.py) | Contour tracing, envelope thickness estimation, and prompt validity checks. |
| [`segmentation.py`](segmentation.py) | Configure MedSAM2, prepare mask prompts, run prompt strategies, and fuse predictions. |
| [`segmentation_util.py`](segmentation_util.py) | Image preprocessing, propagation, prompt combination, and consensus generation. |
| [`evaluation.py`](evaluation.py) | Volume-level and slice-level segmentation metrics. |
| [`sam2/`](sam2/) | SAM2/MedSAM2 model implementation and configurations. |
| [`efficient_track_anything/`](efficient_track_anything/) | Included EfficientTAM implementation. |
| [`training/`](training/) | Included training utilities; not required for the inference example below. |
| [`checkpoints/`](checkpoints/) | Location for separately downloaded model weights. |

## Installation

Use Python **3.10 or newer** in a dedicated environment. The package requires PyTorch **2.5.1 or newer** and torchvision **0.20.1 or newer**. For GPU inference, install a compatible CUDA-enabled PyTorch build for your system using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/JorisMentink/UG-AutoMedSAM2.git
cd UG-AutoMedSAM2

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install PyTorch and torchvision for your system first, then install the repository and notebook dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e .
python -m pip install jupyter matplotlib pandas ipywidgets
```

`setup.py` attempts to build an optional SAM2 CUDA extension. If you need to disable this build, set `SAM2_BUILD_CUDA=0` before running the editable installation (`export SAM2_BUILD_CUDA=0` on Linux/macOS, or `$env:SAM2_BUILD_CUDA="0"` in PowerShell). This can limit extension-dependent post-processing.

Run the notebook or Python examples **from the repository root**, where the project modules and relative checkpoint paths are available.

## Model checkpoint

Download `MedSAM2_latest.pt` using the instructions in the [original MedSAM2 repository](https://github.com/bowang-lab/MedSAM2), then place it at:

```text
checkpoints/MedSAM2_latest.pt
```

The corresponding configuration is `sam2/configs/sam2.1_hiera_t512.yaml`. Pass it to `Segmentation` as `configs/sam2.1_hiera_t512.yaml`, relative to SAM2's configuration package.

Model weights are not included. The `download.sh` mentioned in the checkpoint README is also not included; follow the upstream download instructions instead.

## Data preparation

Data must be obtained separately. The current loader expects the LUND-PROBE directory structure. For prostate CTV inference, each subject needs the following files, relative to the dataset root:

| Relative path | Contents |
| --- | --- |
| `<subject>/MR_StorT2/image.nii.gz` | MRI volume. |
| `<subject>/MR_StorT2/nnUNetOutput/mask_CTVT_427_nnUNet.nii.gz` | Initial nnU-Net segmentation. |
| `<subject>/MR_StorT2/nnUNetOutput/mask_CTVT_427_nnUNet_uncertaintyMap.nii.gz` | Corresponding uncertainty map. |
| `<subject>/MR_StorT2/mask_CTVT_427.nii.gz` | Reference mask; currently loaded unconditionally, including for inference. |
| `<subject>/MR_StorT2/observerData/mask_CTVT_427_step2_obs*.nii.gz` | Observer refinements, required only for observer-based evaluation. |

`parentfolder` should point to the directory containing the subject folders. `subject_nr` is a zero-based index into the alphabetically sorted list of those folders, not a subject identifier.

Arrays use `(slices, height, width)` / `(z, y, x)` order, with spacing in `(z, y, x)` order. Prompt coordinates use `(x, y)` order in the original image space. Images, masks, and uncertainty maps must already be aligned.

For another dataset, adapt the paths in `DataLoader.py` while preserving these conventions. The loader implements prostate (`CTVT`) and rectum paths. Although `bladder` is accepted by its input check, its loading paths are not implemented.

## Run inference

The following example mirrors the two-strategy configuration in the notebook. Replace `parentfolder` with your dataset location.

```python
from DataLoader import DataLoader
from unc_handling import UG_prompter
from segmentation import Segmentation
from segmentation_util import combine_prompt_sets

data = DataLoader(
    parentfolder="/path/to/LUNDPROBE/subjects",
    subject_nr=0,
    volume_of_interest="CTVT",
)

prompter = UG_prompter(data=data)
segmenter = Segmentation(
    data=data,
    checkpoint="checkpoints/MedSAM2_latest.pt",
    cfg="configs/sam2.1_hiera_t512.yaml",
    use_mask_input_as_output_without_sam=True,
    pred_obj_scores=True,
)

prompter.threshold_uncertainty_map(
    unc_threshold=None,
    target_mm=3.0,
    mode="median",
)

staples = prompter.generate_prompts_nietjes(
    unc_band_thr_mm=4.0,
    interpix_dist=5,
    pixel_interval=10,
)
boxes_or_mask = prompter.generate_prompts_boxes(
    band_threshold=3.0,
    interpix_dist=5,
    pixel_interval=10,
)
dense = segmenter.load_dense_prompt()
staples_and_mask = combine_prompt_sets([dense, staples])

segmenter.compile_prompt_sets(
    prompt_dict_list=[staples_and_mask, boxes_or_mask],
    prompt_set_names=["Staple + Coarse", "Bbox + Coarse"],
    prompt_set_weights=[0.4, 0.6],
)

predicted_seg, predicted_logits = segmenter.run_segmentation_sets(
    propagation_style="default",
    weighting_strategy="custom",
    threshold=0.0,
    no_external_propagation=True,
)
```

`predicted_seg` is the binary output array, and `predicted_logits` contains the fused logits. Both are also stored on `segmenter`. Individual strategy outputs are available in `segmenter.segs_per_set` and `segmenter.logits_per_set`.

### Configuration notes

- The weights above apply to the **two complete prompt strategies**: `Staple + Coarse` and `Bbox + Coarse`. They do not weight point prompts against mask prompts within a strategy. Custom weights must sum to 1.
- `generate_prompts_boxes()` uses a bounding box on slices above its thickness threshold and a coarse mask otherwise. The notebook label `Bbox + Coarse` refers to this slice-dependent strategy.
- Envelope thresholding defaults to `method="raycast"`; pass `method="local_normals"` explicitly to use the alternative thickness estimator. Prompt generation has its own method settings.
- `no_external_propagation=True` restricts propagation to the first and last slices containing foreground in the coarse mask. It does not constrain in-plane edits to the uncertainty envelope.
- With mask bypass enabled, mask-only prompts can pass directly to the output. Experiments using continuous Gaussianized mask values should explicitly consider the mask-bypass and object-score settings described in the notebook.

These settings document the current notebook example; they are not a guarantee of reproducing a particular thesis experiment. Record the checkpoint, data split, observer selection, prompt settings, propagation mode, and weights for each experiment.

### Using the notebook

```bash
jupyter notebook LUNDPROBE_inference.ipynb
```

Update `parentfolder` before running the data-loading cell. The notebook currently imports `interactive_inference_viewer` from `plotting_functions.py`, which is not included in this repository. To run inference and evaluation without the viewer, remove that import and skip the final viewer cell. The standalone example above does not require this module.

## Evaluation

To evaluate against a consensus of four observer recontours:

```python
from evaluation import Evaluator

data.load_recontours()
print(data.observer_names)  # Check the observer order before selecting indices.
consensus = data.load_consensus(indices=[0, 1, 2, 3])

metrics = Evaluator(
    pred=predicted_seg,
    gt=consensus,
    spacing=data.img_spacing,
    subject_name=data.subject_name,
).compute_all(surface_dice_tol=1.0)

print(metrics)
```

Consensus generation averages signed distance fields slice by slice, excluding empty observer masks on each slice. Specify the observer indices explicitly: the loader's default selection is `[0, 2, 3]`.

Available metrics include Dice, Hausdorff distance, HD95, mean surface distance, average symmetric surface distance, Surface Dice, centroid distance, and volume differences. `evaluate_slice_by_slice()` provides slice-level evaluation. A separate `compute_contour_within_clinical_boundary()` function evaluates contour agreement with the observer envelope; it assumes four observers and is not included in `Evaluator.compute_all()`.

**Surface Dice implementation:** distances account for voxel spacing, but the score counts boundary voxels rather than weighting by physical surface area. This distinction matters when comparing results with area-weighted Surface Dice implementations, particularly for anisotropic data.

## Acknowledgements and license

This work builds on [MedSAM2](https://github.com/bowang-lab/MedSAM2) and [SAM 2](https://github.com/facebookresearch/sam2), and uses the LUND-PROBE data format. Please acknowledge the original model and dataset authors when using their work.

The repository includes the [Apache License 2.0](LICENSE.txt). Consult the original sources for the terms applicable to separately obtained datasets and model weights.

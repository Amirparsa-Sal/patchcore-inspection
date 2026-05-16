import csv
import logging
import os
import random
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Encode a [0, 1] float anomaly map as a q8rle string.

    Quantizes to [0, 255], flattens column-wise, then run-length encodes
    consecutive identical values. Format:
        q8rle <height> <width> <value_1> <runlen_1> <value_2> <runlen_2> ...
    """
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(
        np.uint8
    )
    h, w = q.shape
    flat = q.T.reshape(-1)
    if flat.size == 0:
        return f"q8rle {h} {w}"
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]
    parts = ["q8rle", str(h), str(w)]
    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]
    return " ".join(parts)


def _segmentation_map_to_2d(segmentation) -> np.ndarray:
    """Collapse leading singleton dimensions so Matplotlib gets a 2D anomaly map."""

    arr = np.asarray(segmentation, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    is_anomaly: Optional[Sequence[bool]] = None,
):
    """Generate anomaly segmentation images and q8rle-encoded prediction CSVs.

    Saves full comparison figures (original, optional ground truth, predicted map)
    under ``segmentation_images/`` for every sample.

    For samples labelled as defective (``is_anomaly`` or inferred from masks),
    the predicted anomaly map alone is also saved under
    ``anomaly_segmentation_maps/`` using the same colour scale ``[0, 1]``.

    Image files and CSV IDs use only the source image basename (no parent
    directory names).

    Predicted anomaly heatmaps use a fixed colour scale ``[0, 1]`` (``vmin`` /
    ``vmax`` on ``imshow``) so maps are comparable across images and runs.

    Writes ``predictions_normal.csv`` and ``predictions_anomalous.csv`` (each
    with columns ``ID``, ``Label``) when sample type is known. If type cannot
    be determined, writes a single ``predictions.csv`` with all rows.

    Args:
        savefolder: [str] Root folder for outputs.
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        is_anomaly: Optional per-image defect flags (same order as *image_paths*).
            When omitted and masks are provided, normal vs anomalous is inferred
            from whether *mask_path* is None (e.g. MVTec ``good`` vs defect).
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    n = len(image_paths)
    if is_anomaly is not None and len(is_anomaly) != n:
        raise ValueError(
            "is_anomaly length {} must match image_paths length {}.".format(
                len(is_anomaly), n
            )
        )

    def per_sample_defect(mask_path):
        """True if sample is treated as anomalous (unknown mask -> False)."""
        if not masks_provided:
            return False
        return mask_path is not None

    images_folder = os.path.join(savefolder, "segmentation_images")
    anomaly_maps_folder = os.path.join(savefolder, "anomaly_segmentation_maps")
    os.makedirs(images_folder, exist_ok=True)

    n_anomaly_maps_saved = 0

    csv_rows_normal = []
    csv_rows_anomalous = []
    csv_rows_all = []

    for idx, (image_path, mask_path, anomaly_score, segmentation) in enumerate(
        tqdm.tqdm(
            zip(image_paths, mask_paths, anomaly_scores, segmentations),
            total=n,
            desc="Generating Segmentation Images...",
            leave=False,
        )
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = os.path.basename(image_path)
        image_id = os.path.splitext(savename)[0]
        label = float_matrix_to_q8rle(segmentation)
        row = (image_id, label)
        csv_rows_all.append(row)

        if is_anomaly is not None:
            defect = bool(is_anomaly[idx])
        elif masks_provided:
            defect = per_sample_defect(mask_path)
        else:
            defect = None

        if defect is True:
            csv_rows_anomalous.append(row)
        elif defect is False:
            csv_rows_normal.append(row)

        savename = os.path.join(images_folder, savename)
        seg2d = _segmentation_map_to_2d(segmentation)
        heatmap_kw = {"vmin": 0.0, "vmax": 1.0}
        if masks_provided:
            f, axes = plt.subplots(1, 3)
            axes[0].imshow(image.transpose(1, 2, 0))
            axes[1].imshow(mask.transpose(1, 2, 0))
            axes[2].imshow(seg2d, **heatmap_kw)
            f.set_size_inches(9, 3)
        else:
            f, axes = plt.subplots(1, 2)
            axes[0].imshow(image.transpose(1, 2, 0))
            axes[1].imshow(seg2d, **heatmap_kw)
            f.set_size_inches(6, 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()

        if defect is True:
            os.makedirs(anomaly_maps_folder, exist_ok=True)
            map_basename = os.path.basename(image_path)
            map_save_path = os.path.join(anomaly_maps_folder, map_basename)
            fig_m, ax_m = plt.subplots(
                1, 1, figsize=(5, 5 * seg2d.shape[0] / max(seg2d.shape[1], 1))
            )
            ax_m.imshow(seg2d, **heatmap_kw, aspect="equal")
            ax_m.set_axis_off()
            fig_m.tight_layout(pad=0)
            fig_m.savefig(map_save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig_m)
            n_anomaly_maps_saved += 1

    if n_anomaly_maps_saved:
        LOGGER.info(
            "Saved %d anomaly-only segmentation maps under %s",
            n_anomaly_maps_saved,
            anomaly_maps_folder,
        )
    if is_anomaly is not None or masks_provided:
        path_normal = os.path.join(savefolder, "predictions_normal.csv")
        path_anomalous = os.path.join(savefolder, "predictions_anomalous.csv")
        for path, rows in (
            (path_normal, csv_rows_normal),
            (path_anomalous, csv_rows_anomalous),
        ):
            with open(path, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["ID", "Label"])
                csv_writer.writerows(rows)
        LOGGER.info(
            "Saved %d normal and %d anomalous predictions to %s and %s",
            len(csv_rows_normal),
            len(csv_rows_anomalous),
            path_normal,
            path_anomalous,
        )
    else:
        csv_path = os.path.join(savefolder, "predictions.csv")
        with open(csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["ID", "Label"])
            csv_writer.writerows(csv_rows_all)
        LOGGER.info(
            "Saved %d predictions to %s (single file; could not split normal vs anomalous)",
            len(csv_rows_all),
            csv_path,
        )


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics

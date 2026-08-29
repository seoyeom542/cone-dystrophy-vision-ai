from pathlib import Path

import numpy as np
import pytest

from src.data import (
    _patient_id,
    build_dataloaders,
    build_display_transform,
    build_transforms,
    split_by_patient,
)


def _samples(patients_per_class: int = 3, scans_per_patient: int = 2):
    samples = []
    for label, class_name in enumerate(("CNV", "DME", "DRUSEN", "NORMAL")):
        for patient in range(patients_per_class):
            for scan in range(scans_per_patient):
                samples.append((f"/{class_name}/{class_name}-{label}{patient:04d}-{scan}.jpeg", label))
    return samples


def test_patient_split_has_no_patient_leakage_and_is_reproducible():
    samples = _samples()
    train_idx, val_idx = split_by_patient(samples, val_ratio=0.34, seed=42)
    assert (train_idx, val_idx) == split_by_patient(samples, val_ratio=0.34, seed=42)

    train_patients = {(label, _patient_id(path)) for path, label in (samples[i] for i in train_idx)}
    val_patients = {(label, _patient_id(path)) for path, label in (samples[i] for i in val_idx)}
    assert train_patients.isdisjoint(val_patients)
    assert {samples[i][1] for i in val_idx} == {0, 1, 2, 3}


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.1])
def test_dataloader_rejects_invalid_val_split(ratio):
    with pytest.raises(ValueError, match="val_split"):
        build_dataloaders(Path("unused"), 32, 2, 0, val_split=ratio)


def test_transforms_produce_matching_square_coordinates():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    tensor = build_transforms(32, train=False)(image=image)["image"]
    display = build_display_transform(32)(image=image)["image"]
    assert tuple(tensor.shape) == (3, 32, 32)
    assert display.shape == (32, 32, 3)

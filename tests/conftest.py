"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import (
    LABEL_HARDHAT,
    LABEL_PERSON,
    LABEL_SAFETY_VEST,
    LABEL_NO_HARDHAT,
    LABEL_NO_SAFETY_VEST,
    Detection,
)


@pytest.fixture()
def person_box() -> Detection:
    """A single person detection centred in a 640×640 frame."""
    return Detection(
        label=LABEL_PERSON,
        confidence=0.9,
        x1=100.0, y1=50.0, x2=300.0, y2=500.0,
    )


@pytest.fixture()
def vest_overlapping() -> Detection:
    """Safety vest that heavily overlaps the person box."""
    return Detection(
        label=LABEL_SAFETY_VEST,
        confidence=0.85,
        x1=110.0, y1=150.0, x2=290.0, y2=350.0,
    )


@pytest.fixture()
def helmet_overlapping() -> Detection:
    """Hardhat that heavily overlaps the top of the person box.

    Person box: (100,50)→(300,500), area=90 000.
    Helmet box: (110,55)→(290,230), area=32 200.
    IoU ≈ 32200/90000 ≈ 0.36 — well above the 0.2 threshold.
    """
    return Detection(
        label=LABEL_HARDHAT,
        confidence=0.80,
        x1=110.0, y1=55.0, x2=290.0, y2=230.0,
    )


@pytest.fixture()
def ppe_far_away() -> Detection:
    """A vest detection far from any person — should not match."""
    return Detection(
        label=LABEL_SAFETY_VEST,
        confidence=0.75,
        x1=400.0, y1=400.0, x2=600.0, y2=600.0,
    )


@pytest.fixture()
def blank_image() -> np.ndarray:
    """640×640 blank white BGR image."""
    return np.full((640, 640, 3), 255, dtype=np.uint8)

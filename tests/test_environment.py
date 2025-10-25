import pytest


def test_numpy() -> None:
    try:
        import numpy as np

        np.zeros(0)
    except ImportError:
        pytest.fail("Numpy is not available")


def test_pandas() -> None:
    try:
        import pandas as pd

        pd.DataFrame()
    except ImportError:
        pytest.fail("Pandas is not available")


def test_scikit_learn() -> None:
    try:
        import sklearn as skl

        skl.show_versions()
    except ImportError:
        pytest.fail("Sci-Kit Learn is not available")


def test_spacy() -> None:
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except ImportError:
        pytest.fail("Spacy is not available")


def test_stylesense() -> None:
    try:
        import stylesense

        stylesense.__name__
    except ImportError:
        pytest.fail("Stylesense module is not available")

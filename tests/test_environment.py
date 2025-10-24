import unittest


class TestImports(unittest.TestCase):
    def test_numpy(self) -> None:
        try:
            import numpy as np

            np.zeros(0)
        except ImportError:
            self.fail("Numpy is not available")

    def test_pandas(self) -> None:
        try:
            import pandas as pd

            pd.DataFrame()
        except ImportError:
            self.fail("Pandas is not available")

    def test_scikit_learn(self) -> None:
        try:
            import sklearn as skl

            skl.show_versions()
        except ImportError:
            self.fail("Sci-Kit Learn is not available")

    def test_spacy(self) -> None:
        try:
            import spacy

            spacy.load("en_core_web_sm")
        except ImportError:
            self.fail("Spacy or en_core_web_sm is not available")


if __name__ == "__main__":
    unittest.main()
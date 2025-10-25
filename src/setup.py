from pathlib import Path

from setuptools import find_packages, setup

cwd = Path(__file__).resolve().parent
requirements = (cwd / "stylesense" / "requirements.txt").read_text().split("\n")

if __name__ == "__main__":
    setup(
        name="stylesense",
        version="1.0.0",
        description="Tools to train the review prediction model",
        packages=find_packages(),
        include_package_data=True,
        package_data={"": ["requirements.txt"]},
        install_requires=requirements,
    )

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

setup(
    name="tetrakis_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "networkx>=3.3,<4.0",
        "numpy>=1.25,<2.0",
    ],
    description="Discrete geometry and wave physics on tetrakis-square lattices",
    long_description=HERE.joinpath("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)

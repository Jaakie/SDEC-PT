from setuptools import setup


setup(
    name="sdecpt",
    version="1.0",
    description="PyTorch implementation of Semi-supervised DEC.",
    author="Jacques van Wyk",
    author_email="jvanwyk40@gmail.com",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=[
        "numpy>=1.13.3",
        "torch>=0.4.0",
        "scipy>=1.0.0",
        "pandas>=0.21.0",
        "visdom>=0.1.05",
        "click>=6.7",
        "xlrd>=1.0.0",
        "cytoolz>=0.9.0.1",
        "tqdm>=4.11.2",
        "scikit-learn>=0.19.1",
        "ptsdae>=1.0.0",
    ],
    packages=["sdecpt"],
)

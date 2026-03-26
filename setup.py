# setup.py
from setuptools import setup, find_packages

setup(
    name="conda_cv",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.13.1",
        "torchvision>=0.14.1",
        "numpy>=1.24.4",
        "opencv-python",
        "Pillow",
        "ultralytics",
        "timm==0.4.9",
        "einops",
        "transformers",
        "pyvis",
        "networkx",
        "cython_bbox"
    ],
    python_requires='>=3.11',
)
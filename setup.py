"""
Setup script for DCAL Twin Face Verification

This package implements the DCAL (Dual Cross-Attention Learning) model
for twin face verification as described in the research paper.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "DCAL Twin Face Verification - A deep learning framework for twin face verification using dual cross-attention learning."

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    # Default requirements if requirements.txt doesn't exist
    requirements = [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.20.0",
        "tensorboard>=2.5.0",
        "mlflow>=2.8.0",  # Local tracking only
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "Pillow>=8.0.0",
        "timm>=0.6.0",
    ]

# Optional dependencies for different features
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "isort>=5.8.0",
        "pre-commit>=2.12.0",
    ],
    "visualization": [
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "streamlit>=1.0.0",
    ],
    "distributed": [
        "horovod>=0.21.0",
        "deepspeed>=0.5.0",
    ],
    "experiment": [
        "tensorboard>=2.5.0",
        "mlflow>=1.18.0",
    ],
    "all": [
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "streamlit>=1.0.0",
        "horovod>=0.21.0",
        "tensorboard>=2.5.0",
        "mlflow>=1.18.0",
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "isort>=5.8.0",
        "pre-commit>=2.12.0",
    ],
}

# Scripts to install
scripts = [
    "scripts/train_twin_verification.py",
    "scripts/evaluate_verification.py",
    "scripts/extract_features.py",
    "scripts/demo.py",
]

setup(
    name="twin-dcal",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="DCAL Twin Face Verification - Dual Cross-Attention Learning for Twin Face Verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/twin-dcal",
    project_urls={
        "Bug Tracker": "https://github.com/example/twin-dcal/issues",
        "Documentation": "https://github.com/example/twin-dcal/docs",
        "Source Code": "https://github.com/example/twin-dcal",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    scripts=scripts,
    entry_points={
        "console_scripts": [
            "dcal-train=scripts.train_twin_verification:main",
            "dcal-evaluate=scripts.evaluate_verification:main",
            "dcal-extract=scripts.extract_features:main",
            "dcal-demo=scripts.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    zip_safe=False,
) 
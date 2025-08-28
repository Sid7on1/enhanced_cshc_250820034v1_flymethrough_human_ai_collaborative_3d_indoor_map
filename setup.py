import os
import sys
from setuptools import setup, find_packages
from typing import List

# Define constants
PROJECT_NAME = "enhanced_cs.HC_2508.20034v1_FlyMeThrough_Human_AI_Collaborative_3D_Indoor_Map"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
DESCRIPTION = "Enhanced AI project based on cs.HC_2508.20034v1_FlyMeThrough-Human-AI-Collaborative-3D-Indoor-Map with content analysis."
LICENSE = "MIT"
URL = "https://github.com/your-username/your-repo-name"

# Define dependencies
INSTALL_REQUIRES: List[str] = [
    "torch",
    "numpy",
    "pandas",
]

# Define development dependencies
EXTRA_REQUIRES: dict = {
    "dev": [
        "pytest",
        "flake8",
        "mypy",
    ],
}

# Define package data
PACKAGE_DATA: dict = {
    "": ["*.txt", "*.md"],
}

# Define entry points
ENTRY_POINTS: dict = {
    "console_scripts": [
        "flyme_through=flyme_through.main:main",
    ],
}

def read_file(filename: str) -> str:
    """Read the contents of a file."""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

def read_requirements(filename: str) -> List[str]:
    """Read the requirements from a file."""
    return read_file(filename).splitlines()

def main() -> None:
    """Main function to setup the package."""
    setup(
        name=PROJECT_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=read_file("README.md"),
        long_description_content_type="text/markdown",
        license=LICENSE,
        url=URL,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        package_data=PACKAGE_DATA,
        entry_points=ENTRY_POINTS,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()
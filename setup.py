# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
setup.py for PyAX25_22.

This script configures the package for installation via pip or setuptools.
It includes all necessary metadata, dependencies, and classifiers for a
production-ready AX.25 v2.2 Python library.
"""

from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the package setup
setup(
    # Package name (PyPI-friendly)
    name="pyax25-22",

    # Version (semantic: major.minor.patch)
    version="0.1.0",

    # Author details
    author="Kris Kirby, KE4AHR",

    # Short description
    description="Pure Python implementation of AX.25 v2.2 Layer 2 protocol for amateur radio applications",

    # Long description from README
    long_description=long_description,

    # Content type for long description
    long_description_content_type="text/markdown",

    # Project URL
    url="https://github.com/ke4ahr/PyAX25_22",

    # Automatically find all packages in the project
    packages=find_packages(),

    # Python version requirement
    python_requires=">=3.8",

    # Runtime dependencies (core requirements)
    install_requires=[
        "pyserial>=3.5",  # For serial KISS interfaces
    ],

    # Optional extras (install with pip install pyax25-22[extra])
    extras_require={
        # For async TCP (AGWPE)
        "async": ["aiohttp>=3.8"],

        # Development tools
        "dev": [
            "pytest>=7.0",       # Testing framework
            "pytest-cov>=4.0",   # Coverage reports
            "black>=24.0",       # Code formatter
            "ruff>=0.1.0",       # Linter
            "mypy>=1.0",         # Static type checking
            "Sphinx>=7.0",       # Documentation generator
            "sphinx-rtd-theme>=2.0",  # Sphinx theme
        ],
    },

    # Classifiers for PyPI (helps with discoverability)
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Communications :: Ham Radio",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",  # Update to 5 - Production/Stable when ready
        "Framework :: Pytest",  # For testing integration
    ],

    # Additional project URLs
    project_urls={
        "Documentation": "https://github.com/ke4ahr/PyAX25_22/tree/main/docs",
        "Bug Reports": "https://github.com/ke4ahr/PyAX25_22/issues",
        "Source": "https://github.com/ke4ahr/PyAX25_22",
    },

    # Entry points for console scripts (if any, e.g., for CLI tools)
    entry_points={
        "console_scripts": [
            # Example: "pyax25-tool = pyax25_22.tools:main",
        ],
    },

    # Keywords for searchability
    keywords="ax25 packet-radio amateur-radio ham-radio kiss agwpe tnc",

    # Include non-code files (e.g., docs, licenses)
    include_package_data=True,

    # MANIFEST.in controls additional files if needed
)

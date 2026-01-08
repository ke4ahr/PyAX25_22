# setup.py
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
setup.py

Configuration for PyAX25_22 distribution.

This file defines the package metadata, dependencies, and classifiers
required for publishing to PyPI and for local installation via pip.

It is fully expanded with current project details as of January 02, 2026.
"""

from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyax25-22",
    version = "0.5.71",
    author="Kris Kirby, KE4AHR",
    description="Pure Python implementation of AX.25 v2.2 Layer 2 protocol for amateur radio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ke4ahr/PyAX25_22",
    project_urls={
        "Bug Tracker": "https://github.com/ke4ahr/PyAX25_22/issues",
        "Documentation": "https://github.com/ke4ahr/PyAX25_22/tree/main/docs",
        "Source Code": "https://github.com/ke4ahr/PyAX25_22",
    },
    packages=find_packages(),
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
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Development Status :: 4 - Beta",
    ],
    keywords="ax25 packet-radio amateur-radio ham-radio kiss agwpe tnc pacsat aprs",
    python_requires=">=3.8",
    install_requires=[
        "pyserial>=3.5",
    ],
    extras_require={
        "async": ["aiohttp>=3.8"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=24.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
            "sphinx>=7.0",
            "sphinx-rtd-theme>=2.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

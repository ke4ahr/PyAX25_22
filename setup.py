# setup.py
from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PyAX25_22",
    version="0.1.0",
    author="Kris Kirby",
    author_email="ke4ahr@example.com",
    description="Python implementation of AX.25 v2.2 protocol suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ke4ahr/PyAX25_22",
    packages=find_packages(include=["pyax25_22", "pyax25_22.*"]),
    install_requires=[
        "pyserial>=3.5",
        "async-timeout>=4.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Communications :: Ham Radio"
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0"],
        "test": ["pytest>=7.0"]
    },
    keywords=[
        "ax25",
        "packetradio",
        "amateurradio",
        "kiss",
        "agwpe"
    ],
    project_urls={
        "Bug Tracker": "https://github.com/ke4ahr/PyAX25_22/issues",
        "Documentation": "https://github.com/ke4ahr/PyAX25_22/wiki"
    },
    license="LGPLv3.0"
)

import io
import os

from setuptools import setup

DESCRIPTION = (
    "Privacy Meter: An open-source library to audit data privacy in statistical and machine learning "
    "algorithms."
)

here = os.path.abspath(os.path.dirname(__file__))
version = 1.0.1
# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name="Privacy-Meter",
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # change to the privacy_meter when the repo is renamed
    url="https://github.com/privacytrustlab/ml_privacy_meter",
    author_email="reza@comp.nus.edu.sg",
    maintainer="Hongyan Chang",
    maintainer_email="hongyan@comp.nus.edu.sg",
    license="MIT",
    packages=["privacy_meter"],
    python_requires=">=3.8.0",
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)

import io
import os
from setuptools import setup

DESCRIPTION = f'Privacy Meter: An open-source library to audit data privacy in statistical and machine learning ' \
              f'algorithms.'

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'project_description.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
      name='privacy_meter',
      version='1.0',
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/privacytrustlab/ml_privacy_meter',
      author_email='reza@comp.nus.edu.sg',
      license="MIT",
      packages=['privacy_meter'],
      python_requires='>=3.6.0',
      include_package_data=True,
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
      ],
)

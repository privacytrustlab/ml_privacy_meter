# Developer Guide 

## Auto-generating Documentation

## Building and Publishing Privacy Meter

We will need a library called `twine` to publish the tool to PyPi. Install this library using `pip install twine` before proceeding with the instructions below.

First, build the tool locally to test if your changes are working as expected. 
Run the following command in a terminal from the root directory (`./privacy_meter/`) to build the local version of the tool:

```bash
pip install -e .
```

You can then import and use the local version of the tool in your test script:

```python
import privacy_meter

# example imports
from privacy_meter.dataset import Dataset
from privacy_meter.model import TensorflowModel
from privacy_meter.information_source import InformationSource
from privacy_meter.audit import Audit, MetricEnum
```

Before building the tool for publishing to PyPi, make the required changes in `./privacy_meter/setup.py`. For example, if you want to update the version number of the tool, you would need to edit the `version` argument passed to the `setup` object. 

Now we can build and upload the tool to PyPi. Run the following commands in a terminal from the root directory:

Create the build files:

```bash
python setup.py sdist bdist_wheel
```

Check if the build files are correct:

```bash
twine check dist/*
```

Upload the files to PyPi:

```bash
twine upload dist/*
```

Note: if you want to upload to `TestPyPi` first, run `twine upload --repository-url https://test.pypi.org/legacy/ dist/*` instead of the command above.
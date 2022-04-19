# Developer Guide 

## Auto-generating Documentation

The tool we use to maintain the documentation is called Sphinx. Install it via pip:

```bash
pip install -U Sphinx
```

We are also using a custom theme, called `sphinx_rtd_theme`. Install it via pip:

```bash
pip install sphinx-rtd-theme
```

### One time configuration

If this is the first time building the documentation on the current machine, run the following command in the root directory of the project:

```bash
sphinx-quickstart
```

Do fill out the basic configuration properties required.

Then, update the `conf.py` file generated inside the `source` folder:
* Update the `html_theme` variable to "sphinx_rtd_theme".
* Update the system path to point to the project’s modules directory (line 13-15)

```python
sys.path.insert(0, os.path.abspath('../privacy_meter/'))
```

* Add "sphinx.ext.autodoc" and "sphinx.ext.napoleon" to the `extensions` list.

Now proceed to the next section.

### Updating the documentation

Start by updating the version number in `/source/conf.py`.

The `/source/index.rst` can be edited manually. You can refer to [this useful cheatsheet](https://sphinx-tutorial.readthedocs.io/cheatsheet/).

We want to automatically generate rst files with autodoc directives from the code:

```bash
sphinx-apidoc -f -o source privacy_meter
```

And now we can generate the html pages from the rst files:

```bash
make html
```

The html files are all contained in the `/build/html/` directory.

Note 1: The test version is then deployed to firebase.

Note 2: This is based on a detailed tutorial for auto-documenting a python project using Sphinx, available at https://betterprogramming.pub/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9

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
from privacy_meter.audit import Audit
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
Installation
============

Whether you're an end-user looking to get started quickly or a developer contributing to the project, this guide outlines the recommended installation methods.

.. contents::
   :local:
   :depth: 1

Normal Installation (End-Users)
-------------------------------

For most users, the recommended way to install the latest **tagged release** of the package is to clone the repository, check out the latest tag, and install it using `pip`:

.. code-block:: bash

   git clone https://github.com/tjrennie/mapext.git
   cd mapext
   git fetch --tags
   git checkout $(git describe --tags `git rev-list --tags --max-count=1`)
   pip install .

Alternatively, you can download the latest tagged release as a ZIP file from the `Releases` page on GitHub, extract it, and install manually:

.. code-block:: bash

   pip install .

.. important::

   This installs the latest stable version based on the most recent Git tag. It is the **recommended approach** for users who want a reliable version of the package without modifying the source code.

Developer Installation
----------------------

If you plan to contribute to the project or make local modifications, install the package in **editable mode**. This allows you to make changes to the code and see them take effect immediately, without reinstalling the package.

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/tjrennie/mapext.git

2. **Navigate to the project directory:**

   .. code-block:: bash

      cd mapext

3. **Install the package in editable mode:**

   .. code-block:: bash

      pip install -e .

4. **(Optional) Install development and documentation dependencies:**

   .. code-block:: bash

      pip install -e .[dev,docs]

   The `[dev,docs]` extras install additional tools required for development, testing, and building documentation.

.. seealso::

   If you're contributing to the package, please follow the guidelines in the :doc:`Contributing Guide <contributing>` for code standards, development workflow, and pull request instructions.

.. _installation:

:github_url: https://github.com/drewmee/PyEEM

************
Installation
************


There are several ways to install the package from source depending on your
local development environment and familiarity with installing python libraries. 
If you are new to python (and don't have `git` installed), install using the
`Install from Source` option.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install directly from pypi (best option)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing directly from pypi using pip is the easiest way forward and will
automagically install any dependencies that aren't already installed.

.. code-block:: shell

    $ pip install pyeem [--upgrade]


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install from GitHub using `pip`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: must have git installed

.. code-block:: shell

    $ pip install --upgrade git+git://github.com/drewmee/PyEEM.git


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clone Repository and Install from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan on contributing to ``pyeem``, you will probably want to fork the
library, and then clone it to your local environment. You can then install from
source directly.

.. code-block:: shell

    $ git clone https://github.com/drewmee/pyeem.git
    $ cd pyeem/
    $ python3 setup.py install


~~~~~~~~~~~~~~~~~~~
Install from Source
~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    $ wget https://github.com/drewmee/pyeem/archive/master.zip
    $ unzip master.zip
    $ cd pyeem-master/
    $ python3 setup.py install

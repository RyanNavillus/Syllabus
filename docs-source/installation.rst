Installation
============

You can install Syllabus on pip with the following command:

.. code-block:: bash
    
        pip install syllabus-rl

To install a development branch of syllabus, you can clone the repository and install it with pip:

.. code-block:: bash

        git clone git@github.com:RyanNavillus/Syllabus.git
        git checkout <branch-name>
        cd Syllabus
        pip install -e .[all]

If you want to modify Syllabus for your own use case or to contribute to the project, you should fork the project and install it in editable mode:

.. code-block:: bash

        git clone git@github.com:<your-username>/<your-syllabus-fork>.git
        cd <your-syllabus-fork>
        pip install -e .[all]
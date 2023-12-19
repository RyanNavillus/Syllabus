.. _TaskSpace:

Task Spaces
===========

Syllabus provides the TaskSpace class as a way to represent the entire space of classes
that are playable in an environment. This is necessary for sampling tasks from the
entire task space.

Usage 
^^^^^
You can define a Discrete task space with 200 tasks as follows:

.. code-block:: python

   from gym.spaces import Discrete
   from syllabus.task_space import TaskSpace

   task_space = TaskSpace(200)
   task_space = TaskSpace(Discrete(200))

Future Features
^^^^^^^^^^^^^^^

This component is currently a work in progress. Future versions will include:

- Mutable Task Spaces

- Train, test, and validation splits over the task space

- Support for more complex task spaces (currently only Discrete, MultiDiscrete, and Box spaces are fully supported)

syllabus.task\_space.task\_space module
---------------------------------------

.. automodule:: syllabus.task_space.task_space
   :members:
   :undoc-members:
   :show-inheritance:

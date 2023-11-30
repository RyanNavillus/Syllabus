.. Syllabus documentation master file, created by
   sphinx-quickstart on Mon Jul 10 07:05:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Syllabus Documentation
======================

Syllabus is a library for using curriculum learning to train reinforcement learning agents. It provides a Curriculum API from
defining curriculum learning algorithms, implementations of popular curriculum learning methods, and a framework for synchronizing 
those curricula across environments running in multiple processes. Syllabus makes it easy to implement curriculum learning methods
and add them to existing training code. It takes only a few lines of code to add a curriculum to an existing training script, and
because of the shared Curriculum API, you can swap out different curriculum learning methods by changing a single line of code.

It currently has support for environments run with Python native multiprocessing or Ray actors, which includes RLLib, CleanRL, 
Stable Baselines 3, and Monobeast (Torchbeast). We currently have working examples with CleanRL, RLLib, and Monobeast (Torchbeast). 
We also have preliminary support and examples for multiagent PettingZoo environments.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   quickstart
   background/curriculum_learning
   curricula/curriculum
   task_spaces/taskspace
   logging
   modules/modules

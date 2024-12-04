.. Syllabus documentation master file, created by
   sphinx-quickstart on Mon Jul 10 07:05:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

Syllabus is a library for using curriculum learning to train reinforcement learning agents. It provides a Curriculum API from
defining curriculum learning algorithms, implementations of popular curriculum learning methods, and a framework for synchronizing 
those curricula across environments running in multiple processes. Syllabus makes it easy to implement curriculum learning methods
and add them to existing training code. It takes only a few lines of code to add a curriculum to an existing training script, and
because of the shared Curriculum API, you can swap out different curriculum learning methods by changing a single line of code.

It currently has support for environments run with Python native multiprocessing or Ray actors, which includes nearly any existing RL library. We have working examples with CleanRL, RLLib, Stable Baselines 3, PufferLib, Moolib, and Monobeast (Torchbeast). Syllabus also supports multiagent PettingZoo environments.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   self
   installation
   quickstart
   motivation
   evaluation/evaluation
   logging
   benchmarks

.. .. toctree::
..    :maxdepth: 1
..    :caption: Curriculum Learning Background:

..    background/curriculum_learning
..    background/ued

.. toctree::
   :maxdepth: 2
   :caption: Curriculum API:

   modules/syllabus.core.curriculum
   curricula/custom_curricula
   curricula/co_player
   curricula/implemented_curricula
   evaluators
   stat_recorder

.. toctree::
   :maxdepth: 1
   :caption: Curriculum Methods:

   curricula/syllabus.curricula.annealing
   curricula/syllabus.curricula.constant
   curricula/syllabus.curricula.domain_randomization
   curricula/syllabus.curricula.expanding_box
   curricula/syllabus.curricula.learning_progress
   curricula/syllabus.curricula.plr
   curricula/syllabus.curricula.selfplay
   curricula/syllabus.curricula.sequential

.. toctree::
   :maxdepth: 1
   :caption: Task Spaces:

   modules/syllabus.task_space
   modules/syllabus.core.task_interface
   modules/syllabus.examples.task_wrappers

.. toctree::
   :maxdepth: 1
   :caption: Synchronization:

   modules/syllabus.core
   synchronization/curriculum_synchronization
   synchronization/environment_synchronization

.. toctree::
   :maxdepth: 2
   :caption: Development:

   Github <https://github.com/RyanNavillus/Syllabus>
   modules/modules

Quickstart
==========

To use Syllabus with your existing training code you need to:

1. Define a :ref:`TaskSpace` for your environment.

2. Choose a Syllabus :ref:`Curriculum API`.

3. Wrap the curriculum with a Syllabus :ref:`synchronization wrapper <Synchronization>`.

4. Wrap your environment with a Syllabus :ref:`TaskWrapper <TaskInterface>`.

^^^^^^^^^^^^^^^^^^^
Define a Task Space
^^^^^^^^^^^^^^^^^^^
The task space represents the range of tasks that you want your curriculum to sample from.
For example, in procgen, the task space is usually a discrete set of 200 seeds.

.. code-block:: python

    from syllabus.task_space import DiscreteTaskSpace 
    task_space = DiscreteTaskSpace(200)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Creating a Syllabus Curriculum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Either use one of the curricula built into Syllabus, or create your own by extending the :ref:`Curriculum <Curriculum>` class.

.. code-block:: python

    from syllabus.curricula import DomainRandomization 
    curriculum = DomainRandomization(task_space)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Synchronizing the Curriculum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use either the native multiprocessing or ray multiprocessing wrapper to synchronize the curriculum across environments.
Syllabus creates a separate multiprocessing channel from your environments, so make sure to choose the same backend (either native or ray).

.. tabs::

   .. tab:: Native Python Multiprocessing

        .. code-block:: python

            from syllabus.core import make_multiprocessing_curriculum
            curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)
   .. tab:: Ray Multiprocessing

        .. code-block:: python

            from syllabus.core import make_ray_curriculum
            curriculum = make_ray_curriculum(curriculum)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Synchronizing the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the matching native multiprocessing or ray multiprocessing wrapper for either gymnasium or pettingzoo to synchronize the environments.

.. tabs::

   .. tab:: Native Python Multiprocessing for Gymnasium Environments

        .. code-block:: python

            from syllabus.core import GymnasiumSyncWrapper
            env = GymnasiumSyncWrapper(env, task_queue, update_queue)


   .. tab:: Native Python Multiprocessing for PettingZoo Environments

        .. code-block:: python

            from syllabus.core import PettingZooSyncWrapper
            env = PettingZooSyncWrapper(env, task_queue, update_queue)

   .. tab:: Ray Multiprocessing for Gymnasium Environments

        .. code-block:: python

            from syllabus.core import RayGymnasiumSyncWrapper
            env = RayGymnasiumSyncWrapper(env)

   .. tab:: Ray Multiprocessing for PettingZoo Environments

        .. code-block:: python

            from syllabus.core import RayPettingZooSyncWrapper
            env = RayPettingZooSyncWrapper(env)

^^^^^^^^
Examples
^^^^^^^^

For more help setting up Syllabus, check out our :ref:`examples <Examples>`  of how to integrate Syllabus with various popular RL libraries. 
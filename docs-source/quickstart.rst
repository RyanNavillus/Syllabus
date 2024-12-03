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
Either use one of the curricula built into Syllabus, or create your own by extending the :ref:`Curriculum API` class.

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
            curriculum = make_multiprocessing_curriculum(curriculum)

   .. tab:: Ray Multiprocessing

        .. code-block:: python

            from syllabus.core import make_ray_curriculum
            curriculum = make_ray_curriculum(curriculum)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Synchronizing the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the matching native multiprocessing or ray multiprocessing wrapper for either gymnasium or pettingzoo to synchronize the environments.

.. tabs::

   .. tab:: Native Multiprocessing for Gymnasium

        .. code-block:: python

            from syllabus.core import GymnasiumSyncWrapper
            env = GymnasiumSyncWrapper(env, curriculum.components)

   .. tab:: Native Multiprocessing for PettingZoo

        .. code-block:: python

            from syllabus.core import PettingZooSyncWrapper
            env = PettingZooSyncWrapper(env, curriculum.components)

   .. tab:: Ray Multiprocessing for Gymnasium

        .. code-block:: python

            from syllabus.core import RayGymnasiumSyncWrapper
            env = RayGymnasiumSyncWrapper(env)

   .. tab:: Ray Multiprocessing for PettingZoo

        .. code-block:: python

            from syllabus.core import RayPettingZooSyncWrapper
            env = RayPettingZooSyncWrapper(env)

^^^^^^^^^^^^^^^^^^
Things to consider
^^^^^^^^^^^^^^^^^^

**Training returns no longer reflect agent performance** - when you use a curriculum, it changes the task distribution in some non-uniform way, often prioritizing easier or harder tasks. This means that training returns no longer reflect the agent's average performance over the task space. You typically need to write a separate evaluation pipeline over a uniform task distribution to properly evaluate agents. You can find more info in the :ref:`Evaluation` section.

**Reward normalization may no longer work** - many baselines in RL will normalize returns automatically using running statistics of the agent's average episodic return. If you are using a curriculum these statistics depend on the task distribution, and may harm the agent's performance by dramatically increasing the nonstationarity of the rewards. If you want to use per-task return normalization, you can use the :ref:`StatRecorder` to track per-task returns.

**Curriculum learning can be slow** - curriculum learning methods do additional computation to select tasks and improve sample efficiency, but this comes at the cost of reduced time efficiency per episode. Syllabus is designed to do this extra computation asynchronously, but it will always be slower than training on a fixed distribution.

**Curriculum learning can change the optimal hyperparameters** - because curriculum learning changes the task distribution, and therefore the reward scale, it can also change the optimal hyperparameters for your agent. You may need to tune your hyperparameters to get the best performance with a curriculum, though you should see some improvement without any tuning if the curriculum works on your environment.

^^^^^^^^
Examples
^^^^^^^^

For more help setting up Syllabus, check out our :ref:`examples <Examples>`  of how to integrate Syllabus with various popular RL libraries. 
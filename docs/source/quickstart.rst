Quickstart
==========

To use Syllabus with your existing training code you need to:

1. Choose a Syllabus :ref:`Curriculum API`.

2. Wrap the curriculum with a Syllabus synchronization wrapper.

3. Wrap your environment with a Syllabus :ref:`TaskWrapper <TaskInterface>`.

4. Define a :ref:`TaskSpace` for your environment.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Creating a Syllabus Curriculum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Either use one of the curricula built into Syllabus, or create your own by extending the Curriculum class.

.. code-block:: python

    from syllabus.curricula import UniformCurriculum 
    curriculum = UniformCurriculum(sample_env.task_space, random_start_tasks=10)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Synchronizing the Curriculum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use either the native multiprocessing or ray multiprocessing wrapper to synchronize the curriculum across environments.
Syllabus creates a separate multiprocessing channel from your environments, so make sure to choose the same backend (either native or ray).

.. code-block:: python

    from syllabus.curricula import UniformCurriculum
    num_envs = 4
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum, num_envs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Synchronizing the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the matching native multiprocessing or ray multiprocessing wrapper to synchronize the environments.

.. code-block:: python

    env = MultiProcessingSyncWrapper(env, task_queue, update_queue)

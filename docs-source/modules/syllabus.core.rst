.. _Synchronization:

Global Synchronization
======================

Syllabus's multiprocessing infrastructure uses a bidirectional sender-receiver model in which the curriculum sends tasks and receives environment metrics, while the environment receives tasks and sends metrics. The environment runs the provided task in the next episode and the curriculum uses the metrics to update its task distribution. You can also update the curriculum directly from the main learner process to incorporate training information. Syllabus is designed to simply work for most environments and curricula, but some complex infrastructure can require more careful configuration. This documentation provides an overview of the synchronization process and common pitfalls to avoid with configuring the synchronization wrappers.


-----------
Basic Usage
-----------

If your environment is vectorized with python multiprocessing (this is true for most libraries, including CleanRL, Stable Baselines 3, and Torchbeast) then you can wrap your curriculum and environment as follows:

.. code-block:: python

   from syllabus.core import make_multiprocessing_curriculum, MultiProcessingSyncWrapper
   curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)
   env = MultiProcessingSyncWrapper(env, task_queue, update_queue)

If your environment is vectorized by Ray (e.g. RLLib) then you can wrap your environment as follows:

.. code-block:: python

   from syllabus.core import make_ray_curriculum, RaySyncWrapper
   curriculum = make_ray_curriculum(curriculum)
   env = RaySyncWrapper(env)

**Note** the environment must be a TaskEnv or wrapped in a TaskWrapper immediately before the synchronization wrapper. When the synchronization wrapper calls the environment's reset method, it must be able to accept a `new_task` keyword argument. You do not need to actually subclass TaskEnv or TaskWrapper, but you must follow the task interface API. See :ref:`Task Interface <TaskInterface>` for more information.

Now that you've applied these wrappers, the environment will automatically receive tasks from the curriculum at the start of each episode, and the environment will automatically send feedback to the curriculum. The environment synchronization wrapper sends updates at the end of each episode, which tell the curriculum to sample another task. The episode return and length may also be used by the ``StatRecorder`` if it is enabled in the curriculum.

You can also update the curriculum directly from the main learner process to incorporate training information. Since this doesn't need multiprocessing synchronization, you can define the interface however you like. However, we suggest making these interfaces as simple as possible to promote portability to other RL libraries. You can read :ref:`Motivation` for more information and see an example of these updates in :ref:`Prioritized Level Replay <prioritized-level-replay-update>`

.. code-block:: python

   curriculum.custom_update(...)


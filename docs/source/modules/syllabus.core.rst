.. _Synchronization:

Global Synchronization
======================

Syllabus's multiprocessing infrastructure uses a bidirectional sender-receiver model in which the curriculum sends tasks and receives environment metrics, while the environment receives tasks and sends metrics. The environment runs the provided task in the next episode and the curriculum uses the metrics to update its task distribution. You can also update the curriculum directly from the main learner process to incorporate training information.


^^^^^
Usage
^^^^^

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

Now that you've applied these wrappers, the environment will automatically receive tasks from the curriculum at the start of each episode. Depending on your environment wrapper options and curriculum method, the curriculum might also receive metrics from the environment at each step or at the end of the episode.

You can also update the curriculum directly from the main learner process to incorporate training information. The exact update metrics will depend on your curriculum method, but the API is the same for all methods. You can see an example of how to do this in various RL libraries using :ref:`Prioritized Level Replay <prioritized-level-replay-update>`

.. code-block:: python

   update = {
      "update_type": "on_demand",
      "metrics": {
         ...
      },
   }
   curriculum.update(update)

syllabus.core.curriculum\_sync\_wrapper module
----------------------------------------------

.. automodule:: syllabus.core.curriculum_sync_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

syllabus.core.environment\_sync\_wrapper module
-----------------------------------------------

.. automodule:: syllabus.core.environment_sync_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

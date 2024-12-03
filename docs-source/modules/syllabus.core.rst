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



---------------------------
Environment Synchronization
---------------------------

The environment synchronization wrappers collect and send the data that will eventually be used to call the ``update_on_step``, ``update_on_episode``, and ``update_on_progress`` methods of the :ref:`Curriculum API`. These wrappers also request a new task at the start of each episode in the ``reset`` method before calling the environment's ``reset`` method. Since updating the curriculum is separate from the main training process, we can batch updates without slowing down training to more efficiently transfer data between processes. Episode and task updates are sent immediately, but step updates are batched for efficient transfer. The size of these batches can be controlled by the ``batch_size`` argument in the synchronization wrapper initializer. Additionally, the environment synchronization wrapper will not send any step updates if the curriculum's ``requires_step_updates`` method is not implemented or returns False.

The environment synchronization wrapper also has a ``buffer_size`` argument, which controls how many tasks will be in the task_queue at any given time. Increasing this can be useful if you want to reduce the amount of time the environment spends waiting for a new task. If the buffer is too small, the environment may run out of tasks and be forced to wait for the curriculum to send more. However, increasing the buffer_size will also cause the environment to be less responsive to changes in the curriculum. Increasing the ``buffer_size`` by one effectively causes the environment to use a task distribution from a number of episodes in the past equal to the number of parallel environments. We have seen some evidence that this delay can impact the performance of methods like PLR, so we recommend setting this value to at most 2. If you are seeing the environment waiting for tasks often, then check if your update queue size is spiking. If it is, you may need to optimize your curriculum because it is having trouble keeping up with the number of updates it is receiving.

Environment Synchronization Wrappers
------------------------------------

.. automodule:: syllabus.core.environment_sync_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

--------------------------
Curriculum Synchronization
--------------------------

The curriculum synchronization wrapper routes update messages to the proper :ref:`Curriculum API` methods and requests tasks from the Curriculum to send back to the environments. You can create a synchronized curriculum by calling the ``make_multiprocessing_curriculum`` or ``make_ray_curriculum`` functions. The ray method will wrap the curriculum in a ray actor, allowing it to communicate with the environments running as separate ray actors. The native multiprocessing method will wrap the curriculum in a wrapper class which has a ``components`` property. This contains all of the information that needs to be passed to the environment synchronization wrapper, including the task and update queues, methods for getting unique environment identifiers, and a method indicating whether the environment should send step updates. Passing these components to the environment synchronization wrapper will allow the curriculum to communicate with the environments. It is possible to have multiple separate curricula and sets of environments communicating via different components but that behavior is not officially supported.

The ``make_multiprocessing_curriculum`` method has several options to provide control over the queue behavior. The ``timeout`` option controls how long each environment will wait for a new task before throwing an error. The remaining communication methods do not block and instead throw an error if a value is not immediately available. Updates are only requested after checking that the update queue is not empty, so this should never throw an error.  Placing updates or tasks into their respective queues will only error if the queue is full. If this happens, you can increase the ``maxsize`` argument to increase the size of both queues, though this is likely indicative of the curriculum processing updates too slowly. If this happens, you can choose to limit the number of environments that are used to update the curriculum with the ``max_envs`` option. This causes the environment synchronization wrapper to only send step updates for at most ``max_envs`` environments. Task updates and environment updates are still sent because they are used to request new tasks from the curriculum, and they are sent at a much slower rate than step updates.


Curriculum Synchronization Wrappers
-----------------------------------

.. automodule:: syllabus.core.curriculum_sync_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Multiagent Curriculum Synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple valid ways to use curricula with multiagent environments. Each agent might be given a separate curriculum, or they may all update the same curriculum. A shared curriculum may be explicitly designed for multiagent environments, or they may be single agent curricula that simply use all of the agent's data to update a single sampling distribution. In cases where the curriculum tracks environments individually, there is also a question of whether each agent from a given environment should be treated as the same data stream, or if each agent should be treated as a separate data stream, similar to a separate single agent environment.

By default, Syllabus sends updates in the PettingZoo format, and therefore assumes that a curriculum is explicitly designed for multiagent environments. If you want to use a single agent curriculum instead, you can wrap the curriculum in the ``MultiagentSharedCurriculumWrapper`` which separates each agent's data and calls the curriculum's ``update_on_step`` method. The ``joint_policy`` argument controls whether the wrapper will separate observations into individual agent observations, or keep the global observation such that a joint policy could select actions for all agents at once, and also creates a unique environment identifier for each agent in each environment. This is only important if you plan to use an `Evaluator <Evaluator>`_ with the observations, or if your curriculum uses the ``env_id`` argument of ``update_on_step``.


Multiagent Synchronization Wrappers
-----------------------------------

.. automodule:: syllabus.core.multiagent_curriculum_wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Additionally, if you have an competitive multiagent, multitask environment and want to design a curriculum over both opponents and tasks, you can use the ``DualCurriculumWrapper`` to combine an agent-based and task-based curriculum into a single curriculum that samples from both. For example, you can use `Self Play <Co-player>`_ with `Prioritized Level Replay <prioritized-level-replay>`_ to individually sample an opponent and seed for each episode. Internally it will send all episode and task updates to both curricula, and it will send step updates to any curriculum that requires them. It will sample tasks from each curriculum and then concatenate them into an ``(agent, task)`` tuple.

Dual Curriculum Wrapper
-----------------------

.. automodule:: syllabus.core.dual_curriculum_wrapper
   :members:
   :undoc-members:
   :show-inheritance:
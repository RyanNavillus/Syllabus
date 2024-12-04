.. _Curriculum-Synchronization:

Curriculum Synchronization
===========================

The curriculum synchronization wrapper routes update messages to the proper :ref:`Curriculum API` methods and requests tasks from the Curriculum to send back to the environments. You can create a synchronized curriculum by calling the ``make_multiprocessing_curriculum`` or ``make_ray_curriculum`` functions. The ray method will wrap the curriculum in a ray actor, allowing it to communicate with the environments running as separate ray actors. The native multiprocessing method will wrap the curriculum in a wrapper class which has a ``components`` property. This contains all of the information that needs to be passed to the environment synchronization wrapper, including the task and update queues, methods for getting unique environment identifiers, and a method indicating whether the environment should send step updates. Passing these components to the environment synchronization wrapper will allow the curriculum to communicate with the environments. It is possible to have multiple separate curricula and sets of environments communicating via different components but that behavior is not officially supported.

The ``make_multiprocessing_curriculum`` method has several options to provide control over the queue behavior. The ``timeout`` option controls how long each environment will wait for a new task before throwing an error. The remaining communication methods do not block and instead throw an error if a value is not immediately available. Updates are only requested after checking that the update queue is not empty, so this should never throw an error.  Placing updates or tasks into their respective queues will only error if the queue is full. If this happens, you can increase the ``maxsize`` argument to increase the size of both queues, though this is likely indicative of the curriculum processing updates too slowly. If this happens, you can choose to limit the number of environments that are used to update the curriculum with the ``max_envs`` option. This causes the environment synchronization wrapper to only send step updates for at most ``max_envs`` environments. Task updates and environment updates are still sent because they are used to request new tasks from the curriculum, and they are sent at a much slower rate than step updates.

.. automodule:: syllabus.core.curriculum_sync_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Multiagent Curriculum Synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are multiple valid ways to use curricula with multiagent environments. Each agent might be given a separate curriculum, or they may all update the same curriculum. A shared curriculum may be explicitly designed for multiagent environments, or they may be single agent curricula that simply use all of the agent's data to update a single sampling distribution. In cases where the curriculum tracks environments individually, there is also a question of whether each agent from a given environment should be treated as the same data stream, or if each agent should be treated as a separate data stream, similar to a separate single agent environment.

By default, Syllabus sends updates in the PettingZoo format, and therefore assumes that a curriculum is explicitly designed for multiagent environments. If you want to use a single agent curriculum instead, you can wrap the curriculum in the ``MultiagentSharedCurriculumWrapper`` which separates each agent's data and calls the curriculum's ``update_on_step`` method. The ``joint_policy`` argument controls whether the wrapper will separate observations into individual agent observations, or keep the global observation such that a joint policy could select actions for all agents at once, and also creates a unique environment identifier for each agent in each environment. This is only important if you plan to use an `Evaluator <Evaluator>`_ with the observations, or if your curriculum uses the ``env_id`` argument of ``update_on_step``.

.. automodule:: syllabus.core.multiagent_curriculum_wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Additionally, if you have an competitive multiagent, multitask environment and want to design a curriculum over both opponents and tasks, you can use the ``DualCurriculumWrapper`` to combine an agent-based and task-based curriculum into a single curriculum that samples from both. For example, you can use `Self Play <Co-player>`_ with `Prioritized Level Replay <prioritized-level-replay>`_ to individually sample an opponent and seed for each episode. Internally it will send all episode and task updates to both curricula, and it will send step updates to any curriculum that requires them. It will sample tasks from each curriculum and then concatenate them into an ``(agent, task)`` tuple.

.. automodule:: syllabus.core.dual_curriculum_wrapper
   :members:
   :undoc-members:
   :show-inheritance:
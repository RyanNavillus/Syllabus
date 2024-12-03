.. _Prioritized Level Replay:

Prioritized Level Replay (PLR) Curriculum
=========================================

Prioritized Level Replay is a simple, yet effective curriculum learning method introduced in https://arxiv.org/pdf/2010.03934.pdf. See this paper for additional information on the method.
The implementation in this code base is based on the original implementation https://github.com/facebookresearch/level-replay/tree/main

PLR has been successfully used to train agents in https://arxiv.org/pdf/2301.07608.pdf with a custom fitness function.

Prioritized Level Replay  samples the next training level by prioritizing those with a higher estimated learning potential. The paper proposes multiple metrics for measuring learning progress, but suggest L1 Value loss or equivalently the Generalized Advantage Estimation (GAE) magnitude as the most effective metric. PLR also utilizes a staleness metric to ensure that every task's learning progress is occasionally updated based on the current policy's capabilities.

In practice prioritized level replay updates it's sampling distribution after each batch, and samples the single highest learning potential task with very high probability. The sampling temperature and task diversity can can be increased by raising the ``temperature`` argument.

The default hyperparameters are tuned for Procgen. When applying PLR to a new environment, you may want to tune the ``staleness_coef``, the replay probability ``rho``, or alter the number of training seeds. You can change the number of training tasks by modifying your task space.


Usage 
^^^^^

PLR expects the environment to be determinstic with respect to the task, which is typically the seed. You may not see good results if your environment is not deterministic for a given task. You can check if your environment is deterministic by modifying the determinism_tests script here `<https://github.com/RyanNavillus/Syllabus/blob/main/tests/determinism_tests.py>`_ to use your environment.

To intialize the curriculum, you will also need to provide the ``num_processes`` which is the number of parallel environments. If you are using Generalized Advantage Estimation, you need to pass the same ``num_steps``, ``gamma``, and ``gae_lambda`` arguments that you use in your training process. You can set any PLR algorithmic options in the ``task_sampler_kwargs_dict``. Please see the :ref:`TaskSampler <TaskSampler>` for a full list of options.

PLR requires L1 Value estimates from the training process to compute it's sampling distirbution, and Syllabus provides several different ways to achieve this, each with its own pros and cons. In short:

* **PrioritizedLevelReplay** - This is the simplest way to add PLR to a project. It receives step updates from the environments and uses an evaluator to recompute the values for each step. This allows you to use it without modifying the training code in any way, but also means it is duplicating a lot of computation.

* **CentralPrioritizedLevelReplay** - This version directly receives value predictions and other data from the training process, and uses them to compute scores.

* **DirectPrioritizedLevelReplay** - This method allows the user to directly provide the scores used in the sampling distribution. It provides the most control over the curriculum, but also has the highest potential for implementation errors.

We recommend using ``PrioritizedLevelReplay`` for initial experiments and tests, then transitioning to ``CentralPrioritizedLevelReplay`` or ``DirectPrioritizedLevelReplay`` for better performance. Since these have higher potential for implementation errors, you can compare their performance against a ``PrioritizedLevelReplay`` baseline to check for discrepancies. Below we go into more detail into how each method operates, and how to configure them for your project.

**Note:** we plan to merge these methods into a single class in the future.

**Note:** the current implementation of ``PrioritizedLevelReplay`` and ``CentralPrioritizedLevelReplay`` only support GAE returns. If you want to use a different return method, you can subclass these methods or use ``DirectPrioritizedLevelReplay``.


Prioritized Level Replay
^^^^^^^^^^^^^^^^^^^^^^^^

This asynchronous implementation of PLR runs automatically with no direct changes to the training code. Once it is configured and the :ref:`synchronization wrappers <Synchronization>` are applied, it will automatically begin sending high-priority tasks to the training environments. ``PrioritizedLevelReplay`` requires an :ref:`Evaluator <Evaluators>` to get the value predictions used to calcualte prioritization scores. This introduces some duplicate computation and in some cases can slow down training, especially in systems where agent inference is the bottleneck. If you need to train agents above 10,000 steps per second, we suggest looking at ``CentralPrioritizedLevelReplay`` or ``DirectPrioritizedLevelReplay``.

The ``buffer_size`` argument to PLR defines how many multiples of ``num_steps`` should be allocated for PLR's buffer. For instance, if the ``num_steps`` is 64 and ``buffer_size`` is 4, then PLR's buffers will hold 256 total steps. PLR needs to hold extra data because in order to efficiently batch value predictions, it needs to evaluate values for all environments at once. However, due to the asynchronous updates, some environments may send multiple batches before other environments send any. This means that PLR may need to hold more than ``num_steps`` steps before it is able to collect values and update the ``TaskSampler``. If one environment is running significantly slower than others, this may lead to an overflow error. If you encounter this issue, you can increase the ``buffer_size`` to hold more steps, or decrease the ``batch_size`` of your environment synchronization wrapper to increase the frequency of updates. Note that the ``batch_size`` argument should never exceed the total buffer size, or the update will fail on the first insert. There are also several warnings and error messages in the code to help you diagnose these issues.

Below is an example of how you can set up ``PrioritizedLevelReplay`` in your project.

.. code-block:: python

   from syllabus.curricula import PrioritizedLevelReplay
   from syllabus.evaluators import CleanRLEvaluator
   from syllabus.core import make_multiprocessing_curriculum, PettingZooSyncWrapper

   # Initialize the environment
   env = Env()

   # Create the Evaluator
   evaluator = CleanRLEvaluator(agent)

   # Initialize the Curriculum
   curriculum = PrioritizedLevelReplay(env.task_space, env.observation_space)
   curriculum = make_multiprocessing_curriculum(curriculum)

   # Wrap the environment
   env = PettingZooSyncWrapper(env, curriculum.components)


For a complete example using ``PrioritizedLevelReplay`` with CleanRL's PPO, see `<https://github.com/RyanNavillus/Syllabus/blob/main/syllabus/examples/training_scripts/cleanrl_procgen.py>`_.

Prioritized Level Replay
------------------------------------------

.. automodule:: syllabus.curricula.plr.plr_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


Central Prioritized Level Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This version of PLR does not require an evaluator but does require additional code to send data from the training loop to the curriculum. Below you can find examples of how to do this for some of the popular RL frameworks.

.. _prioritized-level-replay-update:

.. tabs::

   .. tab:: CleanRL

      Insert the following code at the end of the step loop. For example, `at line 216 in ppo.py <https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py#L216>`_.

      .. code-block:: python

         for step in range(0, args.num_steps):
            ...
         
            with torch.no_grad():
               next_value = agent.get_value(next_obs)
            tasks = [i["task"] for i in infos]

            update = {
               "value": value,
               "next_value": next_value,
               "rew": reward,
               "dones": done,
               "tasks": tasks,
            }
            curriculum.update(update)

   .. tab:: Stable Baselines 3

      You can use a callback to send the values to the curriculum. The callback should be added to the ``learn`` method.

      .. code-block:: python

         class PLRCallback(BaseCallback):
            def __init__(self, curriculum, verbose=0):
               super().__init__(verbose)
               self.curriculum = curriculum

            def _on_step(self) -> bool:
               tasks = [i["task"] for i in self.locals["infos"]]

               obs = self.locals['new_obs']
               obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)
               with torch.no_grad():
                  new_value = self.model.policy.predict_values(obs_tensor)

               update = {
                  "value": self.locals["values"],
                  "next_value": new_value,
                  "rew": self.locals["rewards"],
                  "dones": self.locals["dones"],
                  "tasks": tasks,
               }
               self.curriculum.update(update)
               return True

         curriculum = PrioritizedLevelReplay(task_space)
         model.learn(10000, callback=CustomCallback(curriculum))

   .. tab:: RLLib

      The exact code will depend on your version of RLLib, but you can use callbacks similar to Stable Baselines 3 to update the curriculum after each step `<https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html#rllib-advanced-api-doc>`_.

Central Update Prioritized Level Replay
---------------------------------------------------

.. automodule:: syllabus.curricula.plr.central_plr_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


Direct Prioritized Level Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation of PLR allows you to directly compute your own scores used to prioritize tasks. This gives you the most control over the curriculum, but it can be tricky to implement a good scoring function. Below is an example of how to implement the Value L1 score in CleanRL's PPO. The full script can be found here `<https://github.com/RyanNavillus/Syllabus/blob/main/syllabus/examples/training_scripts/cleanrl_procgen.py>`_.

.. code-block:: python

      a, b = returns.shape
      new_returns = torch.zeros((a + 1, b))
      new_returns[:-1, :] = returns
      new_values = torch.zeros((a + 1, b))
      new_values[:-1, :] = values
      new_values[-1, :] = next_value
      scores = (new_returns - new_values).abs()
      curriculum.update(tasks, scores, dones)

The ``tasks`` and ``dones`` arrays have the shape ``(num_steps, num_envs)`` and the ``scores`` array has the shape ``(num_steps + 1, num_envs)``. We need to expand the size of the value tensor to include the next value prediction, and the returns tensor to match. In some versions of PLR, the next values are also added to the final index of the returns tensor. This effectively removes the next values from the Value L1 score calculation, but allows them to still be used for GAE.


Direct Score Prioritized Level Replay
-----------------------------------------------------------

.. automodule:: syllabus.curricula.plr.direct_plr_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Task Sampler
^^^^^^^^^^^^

The task sampler is shared between the different PLR implementations. It is responsible for calculating and tracking scores, and sampling tasks. It has many different options for sampling strategies that can be configured by passing the ``task_sampler_kwargs`` dictionary to PLR's initializer.

TaskSampler
-------------------------------------------
.. _TaskSampler:

.. automodule:: syllabus.curricula.plr.task_sampler
   :members:
   :undoc-members:
   :show-inheritance:

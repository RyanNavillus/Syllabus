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

To intialize the curriculum, you will also need to provide the ``num-processes`` which is the number of parallel environments. If you are using Generalized Advantage Estimation, you need to pass the same ``num_steps``, ``gamma``, and ``gae_lambda`` arguments that you use in your training process. You can set any PLR algorithmic options in the ``task_sampler_kwargs_dict``. Please see the :ref:`TaskSampler <TaskSampler>` for a full list of options.

PLR requires L1 Value estimates from the training process to compute it's sampling distirbution, and Syllabus provides several different ways to achieve this, each with its own pros and cons. In short:

* PrioritizedLevelReplay - This is the simplest way to add PLR to a project. It receives step updates from the environments and uses an evaluator to recompute the values for each step. This allows you to use it without modifying the training code in any way, but also means it is duplicating a lot of computation.

* CentralPrioritizedLevelReplay - This version directly receives value predictions and other data from the training process, and uses them to compute scores.

* DirectPrioritizedLevelReplay - This method allows the user to directly provide the scores used in the sampling distribution. It provides the most control over the curriculum, but also has the highest potential for implementation errors.

We recommend using PrioritizedLevelReplay for initial experiments and tests, then transitioning to CentralPrioritizedLevelReplay or DirectPrioritizedLevelReplay for better performance. Since these have higher potential for implementation errors, you can compare their performance against a PrioritizedLevelReplay baseline to check for discrepancies. Below we go into more detail into how each method operates, and how to configure them for your project.

**Note** We plan to merge these methods into a single class in the future.


Prioritized Level Replay
^^^^^^^^^^^^^^^^^^^^^^^^


Prioritized Level Replay
------------------------------------------

.. automodule:: syllabus.curricula.plr.plr_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


Central Prioritized Level Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Central Update Prioritized Level Replay
---------------------------------------------------

.. automodule:: syllabus.curricula.plr.central_plr_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


Below you can find examples of how to do this for some of the popular RL frameworks.

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

      The exact code will depend on your version of RLLib, but you can use callbacks similar to Stable Baselines 3 to update the curriculum after each step https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html#rllib-advanced-api-doc.


Direct Prioritized Level Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct Score Prioritized Level Replay
-----------------------------------------------------------

.. automodule:: syllabus.curricula.plr.simple_central_plr_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

TaskSampler
-------------------------------------------
.. _TaskSampler:

.. automodule:: syllabus.curricula.plr.task_sampler
   :members:
   :undoc-members:
   :show-inheritance:

Prioritized Level Replay (PLR) Curriculum
=========================================

Prioritized Level Replay is a simple, yet effective curriculum learning method introduced in https://arxiv.org/pdf/2010.03934.pdf. See this paper for additional information on the method.
The implementation in this code base is based on the original implementation https://github.com/facebookresearch/level-replay/tree/main

PLR has been sucessfully used to train agents in https://arxiv.org/pdf/2301.07608.pdf with a custom fitness function.

Prioritized Level Replay  samples the next training level by prioritizing those with a higher estimated learning potential. The paper proposes multiple metrics for measuring learning progress, but suggest L1 Value loss or equivalently the Generalized Advantage Estimation (GAE) magnitude as the most effective metric. PLR also utilizes a staleness metric to ensure that every task's learning progress is occasionally updated based on the current policy's capabnilities.

In practice prioritized level replay updates it's sampling distribution after each batch, and samples the single highest learning potential task more than 90% of the time. The sampling temperature and task diversity can can be increased by raising the `temperature` argument.

The default hyperparameters are tuned for Procgen. When applying PLR to a new environment, you may want to tune the `staleness_coef`, the replay probability `rho`, or alter the number of training seeds. You can change the number of training tasks by modifying your task space.


Usage 
^^^^^

PLR expects the environment to be determinstic with respect to the task, which is typically the seed. You may not see good results if your environment is deterministic for a given task.

To intialize the curriculum, you will also need to provide the `num-processes` which is the number of parallel environments. We also recommend passing the same `num_steps`, `gamma`, and `gae_lambda` arguments that you use in your training process. You can set any PLR algorithmic options in the `task_sampler_kwargs_dict`. Please see the :ref:`TaskSampler <TaskSampler>` for a full list of options.

PLR requires L1 Value estimates from the training process to compute it's sampling distirbution, so you need to add additional code to your training process to send these values to the curriculum. Below you can find examples of how to do this for some of the popular RL frameworks.

.. tabs::

   .. tab:: CleanRL

      Insert the following code at the end of the step loop. For example, `at line 216 in ppo.py <https://github.com/vwxyzjn/cleanrl/blob/e421c2e50b81febf639fced51a69e2602593d50d/cleanrl/ppo.py#L216>`_.

      .. code-block:: python

         for step in range(0, args.num_steps):
            ...
         
            with torch.no_grad():
               next_value = agent.get_value(next_obs)
            tasks = envs.get_attr("task")

            update = {
               "update_type": "on_demand",
               "metrics": {
                  "value": value,
                  "next_value": next_value,
                  "rew": reward,
                  "dones": done,
                  "tasks": tasks,
               },
            }
            curriculum.update_curriculum(update)
   .. tab:: Stable Baselines 3

      You can use a callback to send the values to the curriculum. The callback should be added to the `learn` method.

      .. code-block:: python

         class PLRCallback(BaseCallback):
            def __init__(self, curriculum, verbose=0):
               super().__init__(verbose)
               self.curriculum = curriculum

            def _on_step(self) -> bool:
               tasks = self.training_env.venv.venv.venv.get_attr("task")

               update = {
                     "update_type": "on_demand",
                     "metrics": {
                        "value": self.locals["values"],
                        "next_value": self.locals["values"],
                        "rew": self.locals["rewards"],
                        "dones": self.locals["dones"],
                        "tasks": tasks,
                     },
               }
               self.curriculum.update_curriculum(update)
               return True

         curriculum = PrioritizedLevelReplay(task_space)
         model.learn(10000, callback=CustomCallback(curriculum))

   .. tab:: RLLib

      The exact code will depend on your version of RLLib, but you can use callbacks similar to Stable Baselines 3 to update the curriculum after each step https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html#rllib-advanced-api-doc.

Prioritized Level Replay
------------------------------------------

.. automodule:: syllabus.curricula.plr.plr_wrapper
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

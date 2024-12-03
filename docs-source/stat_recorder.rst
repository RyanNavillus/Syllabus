.. _StatRecorder:

Stat Recorder
=============

The stat recorder is a utility class that records task-specific episode return and length. It can be used for logging, to track metrics for a custom curriculum, or for per-task reward normalization. If you pass ``record_stats=True`` to your curriculum, a StatRecorder will be automatically be created. Each ``update_on_step`` call will also be passed to the ``StatRecorder`` and each ``log_metrics`` call will also log per-task metrics from the ``StatRecorder``.

If you want to use the stat recorder to normalize rewards for each task, you can call the ``normalize`` method on the ``StatRecorder``. This is particularly useful if you have a curricula over reward functions or environment dynamics, where it is more or less difficult to get rewards in different tasks. Below is an example of how you might normalize rewards with a single environment for simplicity:

.. code-block:: python

    from syllabus.core import StatRecorder
    from syllabus.task_space import DiscreteTaskSpace

    task_space = DiscreteTaskSpace(10)
    curriculum = DomainRandomization(task_space, record_stats=True)

    env = gym.make('procgen:procgen-coinrun-v0')
    obs, info = env.reset()
    episode_return = 0
    episode_length = 0

    while True:
        action = agent.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        episode_return += reward
        episode_length += 1
        normalized_reward = curriculum.stat_recorder.normalize(reward, info['task'])

        if term or trunc:
            curriculum.update_on_episode(episode_return, episode_length, info["task"], 0.0)
            obs, info = env.reset()
            episode_return = 0
            episode_length = 0

Stat Recorder
-----------------------------------

.. automodule:: syllabus.core.stat_recorder
   :members:
   :undoc-members:
   :show-inheritance:
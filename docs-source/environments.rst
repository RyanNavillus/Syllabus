Environment Support
===================

Syllabus is implemented with the new Gymnasium API, which is different from the old OpenAI Gym API.
However, it is possible to use environments implemented with the Gym API in Syllabus.
We recommend using the `Shimmy <https://github.com/Farama-Foundation/Shimmy>`_ package to convert Gym environments to Gymnasium environments.

.. code-block:: python

        import gym 
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

        env = gym.make('CartPole-v0')
        env = GymV21CompatibilityV0(env)


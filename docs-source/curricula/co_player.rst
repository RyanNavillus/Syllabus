.. _Co-player:

Co-player Curricula
===================

**NOTE:** These interfaces may change in the future.

Co-player curricula maintain a distribution of co-players and samples one for the training agent to play with at the start of each episode. These co-players might be partners in cooperative environments, opponents in competitive environments, or a mix of both in mixed-sum games. Fictitious Self-Play, which randomly selects an opponent from a set of previous checkpoints of the protagonist agent, is one example of a curriculum over co-players.

These curricula directly interface with the training agent, and therefore must be called from the main process. For task-based curricula we allow a heterogeneous API for update methods that are called by the training process. However for co-player curricula, we define a homogeneous interface that provide the core functionality to promote interoperability between different co-player curricula. It is of course possible to add additional heterogeneous methods if necessary.

* :mod:`add_agent <syllabus.core.curriculum_base.Curriculum.add_agent>` - adds and agent to the curriculum's agent store. This should be used to provide an efficient interface for saving and loading agents without holding an unbounded number of agents in memory.

* :mod:`get_agent <syllabus.core.curriculum_base.Curriculum.get_agent>` - returns the agent corresponding to the given agent_id. If the agent is already in memory, it simply returns it. If not, it should load it from the disk.


^^^^^
Usage
^^^^^

In practice, the curriculum will assign an agent task to the environment, then the training process will load the corresponding agent for inference. This makes it difficult to efficiently batch inference, because a different policy needs to be loaded for each environment. To avoid this, it may be helpful to batch tasks according to the number of processes in the curriculum's ``sample`` method. This way, the training process will only need to load 2-3 agents at a time, and they will change infrequently.

Here is an example of the intended interaction loop for a single environment. The environment will automatically sample an opponent from the curriculum, and put it's identifier in the info dictionary under the key "task". The training process can then use this identifier to load the corresponding agent:

.. code-block:: python

    from syllabus.core import make_multiprocessing_curriculum, PettingZooSyncWrapper
    from syllabus.curricula import SelfPlay

    agent = MyAgent()
    curriculum = FictitiousSelfPlay()
    curriculum = make_multiprocessing_curriculum(curriculum)

    # Add initial agent to the curriculum
    curriculum.add_agent(agent)

    env = Env()
    env = PettingZooSyncWrapper(env, curriculum.components)
    obs, info = env.reset()

    while True:
        agent_id = info["task"]
        # Get agent for the corresponding task
        agent = curriculum.get_agent(agent_id)

        action = agent.act(obs)
        obs, rew, term, trunc, info = env.step(action)

        if term or trunc:
            obs, info = env.reset()
            agent.learn()

            # Add new agent iteration to the curriculum
            curriculum.add_agent(agent)

We currently have implementations of Self Play, Fictitious Self Play, and Prioritized Fictitious Self Play. You can read more about them here: :ref:`Implemented Curricula`.


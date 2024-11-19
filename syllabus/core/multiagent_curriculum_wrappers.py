from syllabus.core import CurriculumWrapper


class MultiagentSharedCurriculumWrapper(CurriculumWrapper):
    def __init__(self, curriculum, possible_agents, *args, joint_policy=False, **kwargs):
        super().__init__(curriculum, *args, **kwargs)
        self.possible_agents = possible_agents
        self.joint_policy = joint_policy
        self.num_agents = len(possible_agents)

    def update_task_progress(self, task, progress, env_id=None):
        for i in range(self.num_agents):
            self.curriculum.update_task_progress(task, progress, env_id=(env_id * self.num_agents) + i)

    def update_on_step(self, task, obs, reward, term, trunc, info, progress, env_id: int = None) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        for i, agent in enumerate(obs.keys()):
            agent_index = self.possible_agents.index(agent)
            maybe_joint_obs = obs if self.joint_policy else obs[agent]
            env_id = env_id if self.joint_policy else (env_id * self.num_agents) + agent_index
            self.curriculum.update_on_step(task, maybe_joint_obs, reward[i], term[i], trunc[i], info[agent], progress[agent], env_id=env_id)

    def update_on_step_batch(self, step_results, env_id: int = None) -> None:
        tasks, obs, rews, terms, truncs, infos, progresses = step_results
        for t, o, r, te, tr, i, p in zip(tasks, obs, rews, terms, truncs, infos, progresses):
            self.update_on_step(t, o, r, te, tr, i, p, env_id=env_id)

    def update_on_episode(self, episode_return, length, task, progress, env_id = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        for i, agent in enumerate(episode_return.keys()):
            self.curriculum.update_on_episode(episode_return[agent], length, task, progress, env_id=(env_id * self.num_agents) + i)

    def update_batch(self, metrics):
        for update in metrics:
            self.update(update)

    def update(self, metrics):
        update_type = metrics["update_type"]
        args = metrics["metrics"]
        env_id = metrics["env_id"] if "env_id" in metrics else None

        if update_type == "step":
            self.update_on_step(*args, env_id=env_id)
        elif update_type == "step_batch":
            self.update_on_step_batch(*args, env_id=env_id)
        elif update_type == "episode":
            self.update_on_episode(*args, env_id=env_id)
        elif update_type == "on_demand":
            # Directly pass metrics without expanding
            self.curriculum.update_on_demand(args)
        elif update_type == "task_progress":
            self.update_task_progress(*args, env_id=env_id)
        elif update_type == "noop":
            # Used to request tasks from the synchronization layer
            pass
        else:
            raise NotImplementedError(f"Update type {update_type} not implemented.")
        self.unwrapped.n_updates += 1


class MultiagentIndependentCurriculumWrapper(CurriculumWrapper):
    def __init__(self, curriculum, possible_agents, *args, **kwargs):
        super().__init__(curriculum, *args, **kwargs)
        self.possible_agents = possible_agents
        self.num_agents = len(possible_agents)

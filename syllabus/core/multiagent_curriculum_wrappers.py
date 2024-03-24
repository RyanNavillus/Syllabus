from syllabus.core import CurriculumWrapper


class MultiagentSharedCurriculumWrapper(CurriculumWrapper):
    def __init__(self, curriculum, possible_agents, *args, **kwargs):
        super().__init__(curriculum, *args, **kwargs)
        self.possible_agents = possible_agents
        self.num_agents = len(possible_agents)

    def update_task_progress(self, task, progress, env_id=None):
        for i in range(self.num_agents):
            self.curriculum.update_task_progress(task, progress, env_id=(env_id * self.num_agents) + i)

    def update_on_step(self, obs, rew, term, trunc, info, env_id: int = None) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        for i, agent in enumerate(obs.keys()):
            agent_index = self.possible_agents.index(agent)
            self.curriculum.update_on_step(obs[agent], rew[i], term[i], trunc[i], info[agent], env_id=(env_id * self.num_agents) + agent_index)

    def update_on_step_batch(self, step_results, env_id: int = None) -> None:
        obs, rews, terms, truncs, infos = step_results
        for i in range(len(obs)):
            self.update_on_step(obs[i], rews[i], terms[i], truncs[i], infos[i], env_id=env_id)

    def update_on_episode(self, episode_returns, episode_length, episode_task, env_id: int = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        for i, agent in enumerate(episode_returns.keys()):
            self.curriculum.update_on_episode(episode_returns[agent], episode_length, episode_task, env_id=(env_id * self.num_agents) + i)

    def update_batch(self, update_data):
        for update in update_data:
            self.update(update)

    def update(self, update_data):
        update_type = update_data["update_type"]
        args = update_data["metrics"]
        env_id = update_data["env_id"] if "env_id" in update_data else None

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
        elif update_type == "task_progress_batch":
            tasks, progresses = args
            for task, progress in zip(tasks, progresses):
                self.update_task_progress(task, progress, env_id=env_id)
        elif update_type == "add_task":
            self.curriculum.add_task(args)
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

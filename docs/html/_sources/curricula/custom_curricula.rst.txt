
Creating Your Own Curriculum
============================

To create your own curriculum, all you need to do is write a subclass of Syllabus's `Curriculum` class. 
`Curriculum` provides multiple methods for updating your curriculum, each meant for a different context. 
By subclassing the `Curriculum` class, your method will automatically work with all of Syllabus's provided tools and infrastructure.

----------------
Required Methods
----------------

Your curriculum method is REQUIRED to implement the following methods:

* :mod:`sample(k: int = 1) <syllabus.core.curriculum_base.Curriculum.sample>` - Returns a list of `k` tasks sampled from the curriculum.

The `sample` method is how your curriculum decides which task the environments will play.
Most methods use some combination of logic and probability distributions to choose tasks, but there are no restrictions on how you choose tasks.


----------------------------
Curriculum Dependent Methods
----------------------------

Your curriculum will likely require some feedback from the RL training loop to guide its task selection. These might be rewards from the environment, error values from the agent, or some other metric that you define. 
Depending on which type of information your curriculum requires, you will need to implement one or more of the following methods:

* :mod:`update_task_progress(task, progress) <syllabus.core.curriculum_base.Curriculum.update_task_progress>` - is called either after each step or each episode :sup:`1` . It receives a task name and a boolean or float value indicating the current progress on the provided task. Values of True or 1.0 typically indicate a completed task.

* :mod:`update_on_step(obs, rew, term, trunc, info)  <syllabus.core.curriculum_base.Curriculum.update_on_step>` - is called once for each environment step.

* :mod:`update_on_episode  <syllabus.core.curriculum_base.Curriculum.update_on_episode>` - (**Not yet implemented**) will be called once for each completed episode by the environment synchronization wrapper.

* :mod:`update_on_demand(metrics)  <syllabus.core.curriculum_base.Curriculum.update_on_demand>` - is meant to be called by the main learner process to update a curriculum with information from the training process, such as TD errors or gradient norms. It is never used by the individual environments. It receives a dictionary of metrics of arbitrary types.

Your curriculum will probably only use one of these methods, so you can choose to only override the one that you need. For example, the Learning Progress Curriculum
only uses episodic task progress updates with `update_task_progress` and Prioritized Level Replay receives updates from the main process through `update_on_demand`.

:sup:`1` If you choose not to use `update_on_step()` to update your curriculum, set `update_on_step=False` when initializing the environment synchronization wrapper
to prevent it from being called and improve performance (An exception with the same suggestion is raised by default).


-------------------
Recommended Methods
-------------------

For most curricula, we recommend implementing these methods to support convenience features in Syllabus:

* :mod:`_sample_distribution()  <syllabus.core.curriculum_base.Curriculum._sample_distribution>` - Returns a probability distribution over tasks

* :mod:`log_metrics(writer)  <syllabus.core.curriculum_base.Curriculum.log_metrics>` - Logs curriculum-specific metrics to the provided tensorboard or weights and biases logger.

If your curriculum uses a probability distribution to sample tasks, you should implement `_sample_distribution()`. The default implementation of `log_metrics` will log the probabilities from `_sample_distribution()`
for each task in a discrete task space to tensorboard or weights and biases. You can also override `log_metrics` to log other values for your specific curriculum.

----------------
Optional Methods
----------------

You can optionally choose to implement these additional methods:


* :mod:`update_on_step_batch(update_list)  <syllabus.core.curriculum_base.Curriculum.update_on_step_batch>` - Updates the curriculum with a batch of step updates.

* :mod:`update_curriculum_batch(update_data)  <syllabus.core.curriculum_base.Curriculum.update_curriculum_batch>` - Updates the curriculum with a batch of data.


`update_curriculum_batch` and `update_on_step_batch` can be overridden to provide a more efficient curriculum-specific implementation. The default implementation simply iterates over the updates.


Each curriculum also specifies two constants: REQUIRES_STEP_UPDATES and REQUIRES_CENTRAL_UPDATES.

* REQUIRES_STEP_UPDATES - If True, the environment synchronization wrapper should set `update_on_step=True` to provide the curriculum with updates after each step.

* REQUIRES_CENTRAL_UPDATES - If True, the user will need to call `update_on_demand()` to provide the curriculum with updates from the main process. We recommend adding a warning to your curriculum if too many tasks are sampled without receiving updates.

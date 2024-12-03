
Creating Your Own Curriculum
============================

To create your own curriculum, all you need to do is write a subclass of Syllabus's :ref:`Curriculum API` class. It provides multiple methods for updating your curriculum, each meant for a different context. By subclassing the ``Curriculum`` class, your method will automatically work with all of Syllabus's provided tools and infrastructure.

----------------
Required Methods
----------------

Your curriculum is only required to implement a method for sampling tasks:

* :mod:`sample <syllabus.core.curriculum_base.Curriculum.sample>` - Returns a list of ``k`` tasks sampled from the curriculum.

The ``sample`` method is how your curriculum decides which task the environments will use. Most methods use some combination of logic and probability distributions to choose tasks, but there are no restrictions on how you choose tasks.


----------------------------
Curriculum Dependent Methods
----------------------------

Your curriculum will likely require some feedback from the RL training loop to guide its task selection. These might be rewards from the environment, error values from the agent, or some other metric that you define. Depending on which type of information your curriculum requires, you will need to implement one or more of the following methods:

* :mod:`update_on_step <syllabus.core.curriculum_base.Curriculum.update_on_step>` - is called once for each environment step. If you implement this method, you must also implement the ``requires_step_updates`` property to return ``True``. Without this, the environment synchronization wrappers (see :ref:`Synchronization`) will not send step updates to optimize performance.

* :mod:`update_on_episode  <syllabus.core.curriculum_base.Curriculum.update_on_episode>` - will be called once for each completed episode by the environment synchronization wrapper. The default implementation of this method also updates the StatRecorder, so if you override it you should call ``super().update_on_episode()`` to maintain this functionality.

* :mod:`update_task_progress <syllabus.core.curriculum_base.Curriculum.update_task_progress>` - is called either after a task is completed. It receives a task and a boolean or float value indicating the current progress on the provided task. Values of True or 1.0 typically indicate a completed task. If you need to track task progress at each step or each episode you should instead implement ``update_on_step`` or ``update_on_episode`` respectively. This method is used for tasks that complete mid-episode.

Your curriculum will probably only use one of these methods, so you can choose to only override the one that you need. These methods also receive a unique indentifier for the environment that generated the update in case you need to track each environment individually. If your curriculum requires information from the main training process, such as TD errors or gradient magnitudes, you can define your own update method. However, you should make use of the existing update methods as much as possible to minimize the API surface of your method. Updates from the training process have to be implemented differently for each learning library, reducing interoperability. You can look at the different variants of :ref:`Prioritized Level Replay` for examples of how to implement these methods for different libraries.

-------------------
Recommended Methods
-------------------

For most curricula, we recommend implementing these methods to support convenience features in Syllabus:

* :mod:`log_metrics <syllabus.core.curriculum_base.Curriculum.log_metrics>` - Logs curriculum-specific metrics to the provided tensorboard or weights and biases logger.

If your curriculum uses a probability distribution to sample tasks, you should implement ``_sample_distribution()``. The default implementation of ``log_metrics`` will log the probabilities from ``_sample_distribution()`` for each task in a discrete task space to tensorboard or weights and biases. You can also override ``log_metrics`` to log other values for your specific curriculum.

* :mod:`_sample_distribution  <syllabus.core.curriculum_base.Curriculum._sample_distribution>` - Returns a probability distribution over tasks. This is called by ``log_metrics`` to log the sampling distribution. If you don't implement this method, the default implementation will return a uniform distribution over tasks.

-----------------
Co-player Methods
-----------------

If your curriculum is designed to sample over opponent players, you will need to implement these methods:

* :mod:`add_agent <syllabus.core.curriculum_base.Curriculum.add_agent>` - adds and agent to the curriculum's agent store.

* :mod:`get_agent <syllabus.core.curriculum_base.Curriculum.get_agent>` - returns the agent corresponding to the given agent_id.

You can find more information about these methods and co-player curricula here :ref:`Co-player`.

----------------
Optional Methods
----------------

You can optionally choose to implement these additional methods:

* :mod:`update_on_step_batch  <syllabus.core.curriculum_base.Curriculum.update_on_step_batch>` - Updates the curriculum with a batch of step updates. This method receives arrays of values for each input to ``update_on_step`` and can be implemented to provide a more efficient curriculum-specific implementation. The default implementation simply iterates over the updates. The size of the batches is controlled by the batch_size argument of the environment synchronization wrapper initializer.

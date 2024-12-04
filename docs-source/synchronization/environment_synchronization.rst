.. _Environment-Synchronization:

Environment Synchronization
===========================

The environment synchronization wrappers collect and send the data that will eventually be used to call the ``update_on_step``, ``update_on_episode``, and ``update_on_progress`` methods of the :ref:`Curriculum API`. These wrappers also request a new task at the start of each episode in the ``reset`` method before calling the environment's ``reset`` method. Since updating the curriculum is separate from the main training process, we can batch updates without slowing down training to more efficiently transfer data between processes. Episode and task updates are sent immediately, but step updates are batched for efficient transfer. The size of these batches can be controlled by the ``batch_size`` argument in the synchronization wrapper initializer. Additionally, the environment synchronization wrapper will not send any step updates if the curriculum's ``requires_step_updates`` method is not implemented or returns False.

The environment synchronization wrapper also has a ``buffer_size`` argument, which controls how many tasks will be in the task queue at any given time. Increasing this can be useful if you want to reduce the amount of time the environment spends waiting for a new task. If the buffer is too small, the environment may run out of tasks and be forced to wait for the curriculum to send more. However, increasing the ``buffer_size`` will also cause the environment to be less responsive to changes in the curriculum. Increasing the ``buffer_size`` by one effectively causes the environment to use a task distribution from a number of episodes in the past equal to the number of parallel environments. We have seen some evidence that this delay can impact the performance of methods like PLR, so we recommend setting this value to at most 2. If you are seeing the environment waiting for tasks often, then check if your update queue size is spiking. If it is, you may need to optimize your curriculum because it is having trouble keeping up with the number of updates it is receiving.

.. automodule:: syllabus.core.environment_sync_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

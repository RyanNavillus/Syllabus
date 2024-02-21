.. _TaskInterface:

Task Interface
==============

Each environment will define tasks in a different way. A "task" might be the environment seed, an intialization argument, or an entirely different subclass. The `TaskEnv`_ and `TaskWrapper`_ provides a simple interface to change the task of an environment.

A task wrapper defines allows an existing environment to change its task when `reset()` is called. It can be used to map tasks from the task space to actual environment
configurations, or to add entirely new tasks to an environment.
To write a custom task wrapper for an environment, simply subclass the :mod:`TaskWrapper <syllabus.core.task_interface.task_wrapper.TaskWrapper>` for gym environments or :mod:`PettingZooTaskWrapper <syllabus.core.task_interface.task_wrapper.PettingZooTaskWrapper>` for pettingzoo environments.
Syllabus includes a task wrappers for common use cases, as well as examples of task wrappers for specific environments.

----------------
Required Methods
----------------

* :mod:`change_task(task) <syllabus.core.curriculum_base.TaskWrapper.change_task>` - Updates the environment configuration to play the provided task on the next episode.
                                                                                     This gets called before the wrapper's environment is reset.

                                                                                        

If changing the task only requires you to edit properties of the environment, you can do so in the `change_task()` method.
This is called before the internal environment's `reset()` function when you pass a `new_task` to the wrapped environment's `reset()`.
If you need to perform more complex operations, you can also override the entire `reset()` method or other environment methods.

-----------------
Optional features
-----------------



syllabus.core.task_interface.environment\_task\_env module
----------------------------------------------------------

.. _TaskEnv:

.. automodule:: syllabus.core.task_interface.environment_task_env
   :members:
   :undoc-members:
   :show-inheritance:


syllabus.core.task_interface.reinit\_task\_wrapper module
---------------------------------------------------------

.. automodule:: syllabus.core.task_interface.reinit_task_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

syllabus.core.task_interface.subclass\_task\_wrapper module
-----------------------------------------------------------

.. automodule:: syllabus.core.task_interface.subclass_task_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

syllabus.core.task_interface.task\_wrapper module
-------------------------------------------------
.. _TaskWrapper:

.. automodule:: syllabus.core.task_interface.task_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Creating a Task Wrapper for Your Environment
============================================

If you are able to edit your environment

A task wrapper defines allows an existing environment to change its task when `reset()` is called. It can be used to map tasks from the task space to actual environment
configurations, or to add entirely new tasks to an environment.
To write a custom task wrapper for an environment, simply subclass the `TaskWrapper` for gym environments or `PettingZooTaskWrapper` for pettingzoo environments.

----------------
Required Methods
----------------

* :mod:`change_task(task) <syllabus.core.curriculum_base.TaskWrapper.change_task>` - Updates the environment configuration to play the provided task on the next episode.
                                                                                     This gets called before the wrapper's environment is reset.

                                                                                        

If changing the task only requires you to edit properties of the environment, you can do so in the `change_task()` method.
This is called before the internal environment's `reset()` function when you pass a `new_task` to the wrapped environment's `reset()`.
If you need to perform more complex operations, you can also override the `reset()` method or other environment methods.

-----------------
Optional features
-----------------
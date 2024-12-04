.. _TaskSpace:

Task Spaces
===========

Syllabus provides the ``TaskSpace`` class as a way to represent the entire space of classes that are playable in an environment. Task spaces extend the `Gymnasium Spaces <https://gymnasium.farama.org/api/spaces/>`_ by encoding the original task description into a more efficient representation. Encoding tasks provides several benefits:

1. **Interoperability**: Encoding task spaces into their most simple representation makes it easy to implement new curriculum learning methods, without worrying about different possible task encodings.

2. **Ease of Use**: By using a different representation for tasks in the curriculum and the environment, we are able to use the most convenient representation for the environment without affecting the complexity of the curricula.

3. **Optimization**: An efficient encoding makes our multiprocessing synchronization faster and more effective. Specifically, we encode tasks into picklable objects that can be sent between processes.

Similar to Gymnasiumn Spaces, we implement separate ``TaskSpace`` classes for different types of tasks, as well as composite ``TaskSpace`` objects for more complex task spaces.

Usage 
^^^^^
You can define a task space wherever you want, typically in the task wrapper or main training script. You will need to pass this task space to both the environment synchronization wrapper and the curriculum.

.. code-block:: python

   from syllabus.task_space import DiscreteTaskSpace
   task_space = DiscreteTaskSpace(200)    # 200 discrete tasks


Task Space
----------

The TaskSpace is an abstract base class with default behaviors for every task space. If you choose to implement your own TaskSpace class, you should NOT override the ``encode`` or ``decode`` methods directly, because they provide universal type checks and error handling. Instead you should override the ``_encode`` and ``_decode`` methods, which are called by the default implementations of ``encode`` and ``decode``.

.. autoclass:: syllabus.task_space.task_space.TaskSpace
   :members:
   :undoc-members:
   :show-inheritance:

Discrete Task Space
-------------------

The ``DiscreteTaskSpace`` represents a discrete set of tasks. If you do not provide any task names to the intializer, it will use the range of integers from 0 to n-1, where n is the number of tasks. The following are some valid ways to initialize a ``DiscreteTaskSpace``:

.. code-block:: python

   from gymnasium.spaces import Discrete
   from syllabus.task_space import DiscreteTaskSpace
   task_space = DiscreteTaskSpace(200)
   task_space = DiscreteTaskSpace(Discrete(200))
   task_space = DiscreteTaskSpace(200, task_names=[f"task_{i}" for i in range(200)])
   task_space = DiscreteTaskSpace(Discrete(200), task_names=[f"task_{i}" for i in range(200)])

.. autoclass:: syllabus.task_space.task_space.DiscreteTaskSpace
   :members:
   :undoc-members:
   :show-inheritance:

Box Task Space
--------------

The ``BoxTaskSpace`` represents one or more continuous parameters that define a task. It does not encode or decode the task, it simply checks if the task is within the bounds of the task space. The following are some valid ways to initialize a ``BoxTaskSpace``:

.. code-block:: python

   from gymnasium.spaces import Box
   from syllabus.task_space import BoxTaskSpace
   task_space = BoxTaskSpace(Box(low=0.0, high=1.0))
   task_space = BoxTaskSpace(Box(low=[0, 0], high=[1, 1], shape=(2,)))
   task_space = BoxTaskSpace(Box(low=[[0, 0], 0.5, 0.5], high=[[1, 1], [1.5, 1.5]], shape=(2, 2)))

.. autoclass:: syllabus.task_space.task_space.BoxTaskSpace
   :members:
   :undoc-members:
   :show-inheritance:

MultiDiscrete Task Space
------------------------

The ``MultiDiscreteTaskSpace`` represents more than one discrete parameters that define a task. It can either encode each component of the task separately, or if you set ``flatten=True``, it can encode the entire task as a single integer. The following are some valid ways to initialize a ``MultiDiscreteTaskSpace``:

.. code-block:: python

   from gymnasium.spaces import MultiDiscrete
   from syllabus.task_space import MultiDiscreteTaskSpace
   task_space = MultiDiscreteTaskSpace([2, 3, 4])
   task_space = MultiDiscreteTaskSpace(MultiDiscrete([2, 3, 4]))
   task_space = MultiDiscreteTaskSpace([2, 3, 4], [["a", "b"], [0.2, 3.4, -1.1], [5, 6, 7, 8]])
   task_space = MultiDiscreteTaskSpace(MultiDiscrete([2, 3, 4]), [["a", "b"], [0.2, 3.4, -1.1], [5, 6, 7, 8]])

.. autoclass:: syllabus.task_space.task_space.MultiDiscreteTaskSpace
   :members:
   :undoc-members:
   :show-inheritance:

Tuple Task Space
----------------

The ``TupleTaskSpace`` represents a tuple of task spaces. It is useful for representing complex task spaces that are composed of simpler task spaces. It can encode tasks as tuples, where each element of the tuple is encoded by the corresponding task space. You can also set ``flatten=True`` to encode the entire task as a single integer. The following are some valid ways to initialize a ``TupleTaskSpace``:

.. code-block:: python

   from syllabus.task_space import TupleTaskSpace
   task_space = TupleTaskSpace([DiscreteTaskSpace(2), BoxTaskSpace(Box(low=0.0, high=1.0))])
   task_space = TupleTaskSpace([DiscreteTaskSpace(2), BoxTaskSpace(Box(low=0.0, high=1.0))], flatten=True)

**Note:** We do not use the ``Tuple`` space from Gymnasium to initialize the ``TupleTaskSpace``.

.. autoclass:: syllabus.task_space.task_space.TupleTaskSpace
   :members:
   :undoc-members:
   :show-inheritance:
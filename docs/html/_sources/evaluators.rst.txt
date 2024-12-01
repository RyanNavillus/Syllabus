Evaluators
==========

Syllabus provides an evaluator class that allows curricula to interface with the agent throughout training. We provide default implementations for several popular RL libraries and hooks to support most common use cases.
However, it is easy to extend the evaluator class to support new libraries and custom use cases. By default the evaluator holds a reference to the training agent, and can be used to asynchronously get action or value predictions.
If the agent performs differently during training and during evaluation, it is necessary to switch the agent to evaluation mode beforehand. Since the evaluator may be called asynchronously from a separate thread, the Evaluator class
provides a method to copy the agent and periodically synchronize its weights, allowing the copy to be used in evaluation mode while the original agent continues training.

^^^^^^^^^^^^^^^^^^^^^
Creating an Evaluator
^^^^^^^^^^^^^^^^^^^^^
To create an evaluator, simply choose the appropriate evaluator for your RL library and pass it a reference to the agent.

.. code-block:: python

    from syllabus.core.evaluators import CleanRLDiscreteEvaluator
    evaluator = CleanRLDiscreteEvaluator(agent)

Optionally, you can choose to copy the agent to the gpu or cpu for asynchronous evaluation. If your agent has separate training and evaluation behavior, and you choose not to copy the agent, then if the evaluator is used during the training step you may see an error related to training behaviors not being available in evaluation mode.

.. code-block:: python

    from syllabus.core.evaluators import CleanRLDiscreteEvaluator
    evaluator = CleanRLDiscreteEvaluator(agent, copy_agent=True, device="cpu")

Evaluators
----------

.. automodule:: syllabus.core.evaluator
   :members:
   :undoc-members:
   :show-inheritance:
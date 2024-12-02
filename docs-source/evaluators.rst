Evaluators
==========

Syllabus provides an evaluator class that allows curricula to interface with the agent throughout training. We provide default implementations for several popular RL libraries and hooks to support most common use cases. However, it is easy to extend the evaluator class to support new libraries and custom use cases. By default the evaluator holds a reference to the training agent, and can be used to asynchronously get action or value predictions. If the agent performs differently during training and during evaluation, it is necessary to switch the agent to evaluation mode beforehand. Since the evaluator may be called asynchronously from a separate thread, the Evaluator class provides a method to copy the agent and periodically synchronize its weights, allowing the copy to be used in evaluation mode while the original agent continues training.

^^^^^^^^^^^^^^^^^^^^^
Creating an Evaluator
^^^^^^^^^^^^^^^^^^^^^
To create an evaluator, simply choose the appropriate evaluator for your RL library and pass it a reference to the agent.

.. code-block:: python

    from syllabus.core.evaluators import CleanRLEvaluator
    evaluator = CleanRLEvaluator(agent)

Optionally, you can choose to copy the agent to the gpu or cpu for asynchronous evaluation. If your agent has separate training and evaluation behavior, and you choose not to copy the agent, then if the evaluator is used during the training step you may see an error related to training behaviors not being available in evaluation mode.

.. code-block:: python

    from syllabus.core.evaluators import CleanRLEvaluator
    evaluator = CleanRLEvaluator(agent, copy_agent=True, device="cpu")

If your RL library preprocesses observations between the environment and the agent, you may need to preprocess the observations before passing them to the evaluator. You can do this by passing a ``_preprocess_obs`` callback to your evaluator.

.. code-block:: python

    from syllabus.core.evaluators import CleanRLEvaluator

    def _preprocess_obs(obs):
        # Preprocess the observation
        return obs

    evaluator = CleanRLEvaluator(agent, preprocess_obs=_preprocess_obs)

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implementing a Custom Evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To implement an evaluator, you will need to convert the inputs and outputs to match the format expected by the agent for the ``_get_action``, ``_get_value``, and ``_get_action_and_value`` methods. The ``_get_action_and_value`` method can be implemented to avoid duplicate computations when both actions and values are needed. Otherwise, you can leave the default implementation which simply calls both ``_get_action`` and ``_get_value`` separately.

Note that you should NOT override the ``get_action``, ``get_value``, or ``get_action_and_value`` methods directly, because they wrap your custom ``_get_action``,  ``_get_value``, and ``_get_action_and_value`` methods in logic that avoids common implementation mistakes and raises warnings for common user errors. More specifically, these methods perform the following operations in order:

1. Update the copied agent's weights to match the training agent

2. Format observations using the evaluator's ``_prepare_state`` method. The default implementation of this method calls the user provided ``_preprocess_obs`` method first, if it is provided.

3. Format lstm states using the evaluator's ``_prepare_lstm`` method.

4. Set the agent to evaluation mode.

5. Call ``_get_action``, ``_get_value``, or ``get_action_and_value`` wrapped in a ``torch.no_grad()`` scope.

6. Set the agent to training mode.

If there is additional information from the agent that you would like to use in your curriculum, you can return it in the ``extras`` dictionary of the output.

.. code-block:: python

    from syllabus.core.evaluator import Evaluator
    class CustomEvaluator(Evaluator):
        def _get_action(self, inputs):
            # Convert inputs to the format expected by the agent
            # Call the agent's get_action method
            # Convert the output to the format expected by the curriculum
            return action, lstm_state, extras

        def _get_value(self, inputs):
            # Convert inputs to the format expected by the agent
            # Call the agent's get_value method
            # Convert the output to the format expected by the curriculum
            return value, lstm_state, extras

        def _get_action_and_value(self, inputs):
            # Convert inputs to the format expected by the agent
            # Call the agent's get_action_and_value method
            # Convert the output to the format expected by the curriculum
            return action, value, lstm_state, extras

        def _prepare_state(self, state):
            # Convert the state to the format expected by the agent.
            # Your method should start by calling ``self._preprocess_obs`` if it is provided.
            return state

        def _prepare_lstm(self, lstm_state, done):
            # Convert the lstm state to the format expected by the agent
            return lstm_state, done

        def _set_eval_mode(self):
            # Set the agent to evaluation mode
            pass

        def _set_train_mode(self):
            # Set the agent to training mode
            pass


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using an Evaluator in your Curriculum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use the evaluator in your curriculum, simply call the evaluator's ``get_action``, ``get_value``, or ``get_action_and_value`` methods. You'll need to track the inputs within your curriculum, which typically includes the observations and lstm state of the agent.
If you need to generate values for an entire trajectory, be careful to evaluate states sequentially for lstm agents. The evaluator takes care of all the necessary steps to ensure that the agent is in the correct mode and that the inputs are formatted correctly. As long as you format the inputs and read the outputs correctly, you can use the evaluator in your curriculum without worrying about the details of the agent's implementation.

Evaluators
----------

.. automodule:: syllabus.core.evaluator
   :members:
   :undoc-members:
   :show-inheritance:
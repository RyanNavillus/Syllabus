Learning Progress
=================

This is an implementation of the curriculum introduced in the paper `Multi-task curriculum learning in a complex, visual,
hard-exploration domain: Minecraft <https://arxiv.org/pdf/2106.14876.pdf>`_ (Kanitscheider et al 2021). It has been used to achieve strong performance in minecraft without offline data.
The method tracks the completion rate of individual binary tasks over time. It maintains two exponential moving averages, one fast and one slow, of this task progress over the course of training. By measuring the difference between these two moving averages, we can determine whether the agent is making recent progress on a task. If the difference is positive, the agent is learning to solve the task. If the difference is negative, the agent is forgetting how to solve a task. To improve performance in both cases, the curriculum samples tasks according to the magnitude of the learning progress. You can reference the paper for more details.

.. automodule:: syllabus.curricula.learning_progress
   :members:
   :undoc-members:
   :show-inheritance:

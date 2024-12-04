Learning Progress
=================

This is an implementation of the curriculum introduced in the paper `Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft <https://arxiv.org/pdf/2106.14876.pdf>`_ (Kanitscheider et al 2021). It has been used to achieve strong performance in minecraft without offline data.It maintains a fast and slow exponential moving average (EMA) of the task completion rates for a set of discrete tasks. By measuring the difference between the fast and slow EMAs and reweighting it to adjust for the time delay created by the EMA, this method can estimate the learning progress of a task. If the difference is positive, the agent is learning to solve the task. If the difference is negative, the agent is forgetting how to solve a task. To improve performance in both cases, the curriculum samples tasks according to the magnitude of the learning progress. You can reference the paper for more details. Syllabus's implementation is based on the open-source implementation used for `OMNI <https://arxiv.org/abs/2306.01711>`_ that can be found here: `<https://github.com/jennyzzt/omni>`_.

.. automodule:: syllabus.curricula.learning_progress
   :members:
   :undoc-members:
   :show-inheritance:

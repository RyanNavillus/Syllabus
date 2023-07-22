.. _Curriculum API:

Curriculum
==========

Syllabus's Curriculum API is a unified interface for curriculum learning methods. Curricula following this API
can be used with all of Syllabus's infrastructure. We hope that future curriculum learning research will provide
implementations following this API to encourage reproducibility and ease of use.

The full documentation for the curriculum class can be found :doc:`../syllabus.core`

The Curriculum class has three main jobs:

- Maintain a sampling distribution over the task space.

- Incorporate feedback from the environments or training process to update the sampling distribution.

- Provide a sampling interface for the environment to draw tasks from.


In reality, the sampling distribution can be whatever you want, such as a uniform distribution,
a deterministic sequence of tasks, or a single constant task depending on the curriculum learning method.

To incorporate feedback from the environment, the API provides multiple methods:

- :mod:`update_on_step <syllabus.core.curriculum_base.Curriculum.update_on_step>`

- :mod:`update_task_progress <syllabus.core.curriculum_base.Curriculum.update_task_progress>`

- :mod:`update_on_episode <syllabus.core.curriculum_base.Curriculum.update_on_episode>`

- :mod:`update_on_step_batch <syllabus.core.curriculum_base.Curriculum.update_on_step_batch>`

- :mod:`update_curriculum_batch <syllabus.core.curriculum_base.Curriculum.update_curriculum_batch>`


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   custom_curricula

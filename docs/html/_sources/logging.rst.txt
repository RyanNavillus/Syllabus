Logging
=======

Syllabus currently support logging through TensorBoard or Weights and Biases through the :mod:`log_metrics <syllabus.core.curriculum_base.Curriculum.log_metrics>` function for curricula. This method collects logs and passes them along to the base Curriculum class, which performs the actual logging. As such, you must call `super().log_metrics()`` in your custom curriculum's log_metrics function to ensure that the logs are propagated and saved.

.. autofunction:: syllabus.core.curriculum_base.Curriculum.log_metrics
    :noindex:
.. _Motivation:

Motivation
==========

Syllabus was created to help researchers to study curriculum learning algorithms on complex environments. Challenging environments like Chess, Go, and Starcraft have served as milestones for progress in AI. However, complex problems often require complex solutions, so the best agents in many RL environments utilize custom learning infrastructure with task-specific methods. Conversely, new methods are often introduced as standalone libraries, with support for a limited number of algorithms and environments. This makes it extremely difficult to apply new methods to complex problems. As a consequence of these challenges, the most impressive applications of reinforcement learning have been achieved by large industry labs with extensive manpower and computational resources. In particular, curriculum learning is a nearly ubiquitous component of these achievements, but it is rarely used in academic research. Syllabus aims to change this by providing a portable and flexible API for curriculum learning that can be easily integrated into existing training code. This allows researchers to experiment with curriculum learning methods on complex environments without needing to build custom infrastructure, thereby mitigating implementation errors and promoting reproducibility.

Curriculum learning is a unique class of methods in that they only modify the task distribution of the environment, not the core learning process. It is therefore possible to implement curricula as modular components that can be easily added to existing training code. Syllabus provides an API for defining curricula which encourages the minimum possible interface surface area, meaning less work for practitioners and fewer opportunities for implementation errors. This is in contrast to existing curriculum learning libraries, which are implemented as standalone solutions and do not intergrate with existing RL libraries.

Many of the major RL successes on challenging problems have used curriculum learning, and challenging academic benchmarks have been developed as testbeds for autocurricula. Despite this, most academic research focuses on simple environments like Procgen and Minigrid. Although these benchmarks have led to many important discoveries, it is clear that many of the methods and lessons from these environments do not scale to more difficult problems. Syllabus aims to bridge this gap by providing a framework for studying curriculum learning on complex environments.


^^^^^^^^^^^^
Design Goals
^^^^^^^^^^^^

Guided by these goals, Syllabus was designed with several focuses in mind:

1. Integrating Syllabus into training infrastructure should require minimal code changes.

2. Code complexity should scale with the complexity of the curriculum learning algorithm.

3. Syllabus should be general to support the many different formulations of curriculum learning.

4. Algorithm logic should be contained in the minimum possible number of files.

Syllabus's main goal is to improve the ease of use and reproducibility of curriculum learning methods. Both of these benefit from interfaces with small surface areas, and library agnostic code. The fewer changes that need to be made to existing training code, the fewer opportunities there are for implementation errors. Focusing on interoperability minimizes the work that practitioners need to do to use curriculum learning in their work.

To make this library widely useful for the community, we tried to distill the shared structure of curriculum learning algorithms into a single API. This makes it easier to add a curriculum to Syllabus than it is to write from scratch. However, curriculum learning methods can have extremely varied components, so in cases where Syllabus does not provide a required feature, we recommend heterogeneous APIs that add on top of Syllabus's existing interfaces  This approach also allows Syllabus to grow naturally with new use cases by adding APIs and improving the portability of new autocurricula over time.

Finally, inspired by the success of `CleanRL <https://github.com/vwxyzjn/cleanrl>`_ and other single-file libraries, Syllabus's Curriculum API encourages the use of single or few-file implementations of algorithms. This approach has been shown to speed up prototyping and improve reproducibility, because complex object-oriented systems tend to obfuscate important implementation details. However, Syllabus's infrastructure also needs to be optimized to support many possible RL libraries and applications. Syllabus reaches a middle ground by providing well-engineered infrastructure that interfaces with simple, flat curriculum implementations. This places the core logic of curriculum learning methods in a single file, separate from the reinforcement learning code, while allowing the algorithm-agnostic infrastructure to be carefully tested and optimized.

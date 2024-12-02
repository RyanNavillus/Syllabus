Motivation
==========

Syllabus was created to help researchers to study curriculum learning algorithms on complex environments. Challenging environments like Chess, Go, and Starcraft have served as milestones for AI and working towards solving these benchmarks is the main driver of progress in RL. However, complex problems often require complex solutions, so the best agents in many RL environments utilize custom learning infrastructure. This makes it extremely difficult to apply new methods to these problems. As a result, the most impressive applications of reinforcement learning have been achieved by large industry labs with extensive manpower and computational resources. Curriculum learning is a nearly ubiquitous component of these achievements, but it is rarely used in academic research. Syllabus aims to change this by providing a portable and flexible API for curriculum learning that can be easily integrated into existing training code. This allows researchers to experiment with curriculum learning methods on complex environments without needing to build custom infrastructure mitigating any possibility of implementation error and promoting reproducibility.

Curriculum learning is a unique approach in that it only modifies the task distribution of the environment, not the core learning process. It is therefore possible to implement curricula as modular components that can be easily added to existing training code. This is in contrast to other curriculum learning libraries, which often come as standalone learning libraries. Syllabus provides an API for defining curricula which encourages the minimum possible interface surface area, meaning less work for practitioners and fewer opportunities for implementation errors.

Many of the major RL successes on challenging problems have used curriculum learning. Since then, challenging academic benchmarks have been developed as testbeds for autocurricula. Despite this, most academic research often focuses on simple environments like Procgen and Minigrid. Although these benchmarks have led to many important discoveries, it is clear that many of our methods and findings do not scale to more difficult problems. Syllabus aims to bridge this gap by providing a framework for studying curriculum learning on complex environments.


^^^^^^^^^^^^
Design Goals
^^^^^^^^^^^^

Guided by these goals, Syllabus was designed with several focuses in mind:

1. Integrating Syllabus into training infrastructure should require minimal code changes.

2. Code complexity should scale with the complexity of the curriculum learning algorithm.

3. Syllabus should be general to support the many different formulations of curriculum learning.

4. Algorithm logic should be contained in the minimum possible number of files.

Syllabus's main goal is to improve the ease of use and reproducibility of curriculum learning methods. Both of these benefit from interfaces with small surface areas, and library agnostic code. The fewer changes that need to be made to existing training code, the fewer opportunities there are for implementation errors. Focusing on interoperability minimizes the work that practitioners need to do to use curriculum learning in their work.

To make this library widely useful for the community, we tried to distill the shared components of curriculum learning algorithms into a single API. Curriculum learning methods can have extremely varied components, so in cases where we don't implement a required feature, we allow for heterogeneous APIs that add on top of the features provided by Syllabus rather than replacing them. This makes it at least as easy to add a curriculum to Syllabus than it is to write from scratch, and typically makes it much easier. This approach also allows Syllabus to grow naturally with new use cases, adding new APIs to provide portability to new classes of autocurricula over time.

Finally, inspired by the success of `CleanRL <https://github.com/vwxyzjn/cleanrl>`_ and other single-file libraries, Syllabus's Curriculum API supports the use of single or few-file implementations of algorithms. This approach has been shown to speed up prototyping and improve reproducibility, because complex object-oriented systems tend to obfuscate important implementation details. Syllabus reaches a middle ground by providing well-engineered and tested infrastructure that interfaces with flat curriculum implementations. This allows researchers to focus on the core logic of their curriculum learning and reinforcement learning methods, while still benefiting from the shared infrastructure of Syllabus.

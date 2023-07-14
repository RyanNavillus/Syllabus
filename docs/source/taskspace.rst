Task Spaces
============

Syllabus provides the TaskSpace class as a way to represent the entire space of classes
that are playable in an environment. This is necessary for sampling tasks from the
entire task space.

This component is currently a work in progress. Future versions will include:
- Mutable Task Spaces
- the ability to define train, test, and validation splits over the task space
- Support for more complex task spaces (currently only Discrete and Box spaces are fully supported)
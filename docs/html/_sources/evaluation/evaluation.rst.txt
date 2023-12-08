Evaluation
==========

Evaluating RL agents trained with curriculum learning requires special consideration. Typically training tasks are assumed to be drawn from the same distribution as the test tasks. However, curriculum learning methods modify the training task distribution to improve test performance. Therefore, training returns are not a good measure of performance. Agents shoudl be periodically evaluated during training on uniformly sampled tasks, ideally from a held out test set. You can see an example of this approach in  .

Correctly implementing this evaluation code can be surprisingly challenging, so we list a few guidelines to keep in mind here:
* Make sure not to bias evaluation results towards shorter episodes. This is is easy to do by accident if you try to multiprocess evaluations. For example, if you run a vectorized environment and save the first 10 results, your test returns will be biased toward shorter episodes, which likely earned lower returns.
* Reset the environments before each evaluation. This may seem obvious, but if you since some vectorized environments don't allow you to directly reset the environments, some might be tempted to skip this step.
* Use the same environment wrappers for the evaluation environment. This is important because some wrappers, such as the `TimeLimit` wrapper, can change the dynamics of the environment. If you use different wrappers, you may get different results.
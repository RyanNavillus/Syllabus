import unittest

import numpy as np

from syllabus.curricula import SimulatedAnnealing
from syllabus.tests import create_cartpole_env


class TestAnnealingBoxCurriculum(unittest.TestCase):

    def test_annealing_values(self):
        cartpole_env = create_cartpole_env()
        curriculum = SimulatedAnnealing(
            task_space=cartpole_env.task_space,
            start_values=[0.2, 0.5],
            end_values=[0.8, 0.9],
            total_steps=[10, 10]
        )

        # Expected values
        expected_values = [
            [0.20, 0.50],
            [0.26, 0.54],
            [0.32, 0.58],
            [0.38, 0.62],
            [0.44, 0.66],
            [0.50, 0.70],
            [0.56, 0.74],
            [0.62, 0.78],
            [0.68, 0.82],
            [0.74, 0.86],
            [0.80, 0.90]
        ]

        for i in range(11):
            sampled_values = curriculum.sample(k=1)[0]
            curriculum.update_on_step()  # Simulate the update on step
            np.testing.assert_almost_equal(sampled_values, expected_values[i], decimal=3)

    def test_negative_start_end_values(self):
        cartpole_env = create_cartpole_env()
        curriculum = SimulatedAnnealing(
            task_space=cartpole_env.task_space,
            start_values=[-0.2, -0.5],
            end_values=[-0.8, -0.9],
            total_steps=10
        )

        expected_values = [
            [-0.20, -0.50],
            [-0.26, -0.54],
            [-0.32, -0.58],
            [-0.38, -0.62],
            [-0.44, -0.66],
            [-0.50, -0.70],
            [-0.56, -0.74],
            [-0.62, -0.78],
            [-0.68, -0.82],
            [-0.74, -0.86],
            [-0.80, -0.90]
        ]

        for i in range(11):
            sampled_values = curriculum.sample(k=1)[0]
            curriculum.update_on_step()
            np.testing.assert_almost_equal(sampled_values, expected_values[i], decimal=3)

    def test_reverse_annealing(self):
        cartpole_env = create_cartpole_env()
        curriculum = SimulatedAnnealing(
            task_space=cartpole_env.task_space,
            start_values=[0.7, 0.7],
            end_values=[0.1, -0.1],
            total_steps=10
        )

        expected_values = [
            [0.70, 0.70],
            [0.64, 0.62],
            [0.58, 0.54],
            [0.52, 0.46],
            [0.46, 0.38],
            [0.40, 0.30],
            [0.34, 0.22],
            [0.28, 0.14],
            [0.22, 0.06],
            [0.16, -0.02],
            [0.10, -0.10]
        ]

        for i in range(11):
            sampled_values = curriculum.sample(k=1)[0]
            curriculum.update_on_step()
            np.testing.assert_almost_equal(sampled_values, expected_values[i], decimal=3)


if __name__ == '__main__':
    unittest.main()

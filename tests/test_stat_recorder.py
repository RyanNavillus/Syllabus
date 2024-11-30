import gymnasium as gym
import numpy as np
from collections import deque
from syllabus.task_space import DiscreteTaskSpace
from syllabus.core import StatRecorder


def test(stat_recorder, simulated_eps_info, expected_episode_reward_mean_by_task, expected_episode_reward_std_by_task, expected_episode_length_mean_by_task, expected_episode_length_std_by_task, expected_normalized_reward):
    num_passed = 0
    num_failed = 0
    for i in range(len(simulated_eps_info)):
        task_id, eps_reward, eps_length = simulated_eps_info[i]
        stat_recorder.record(eps_reward, eps_length, task_id)
        rewards_for_task = deque([x[1] for x in simulated_eps_info[:i+1] if x[0] == task_id], maxlen=10)
        normalized_reward_for_task = stat_recorder.normalize(rewards_for_task, task_id)
        try:
            assert abs(stat_recorder.stats[task_id]['mean_r'] - expected_episode_reward_mean_by_task[i]) < 1e-7
            assert abs(np.sqrt(stat_recorder.stats[task_id]['var_r']) - expected_episode_reward_std_by_task[i]) < 1e-7
            assert abs(stat_recorder.stats[task_id]['mean_l'] - expected_episode_length_mean_by_task[i]) < 1e-7
            assert abs(np.sqrt(stat_recorder.stats[task_id]['var_l']) - expected_episode_length_std_by_task[i]) < 1e-7
            for j in range(len(normalized_reward_for_task)):
                assert abs(normalized_reward_for_task[j] - expected_normalized_reward[i][j]) < 1e-7
            print(f"Test case {i} PASSED.")
            num_passed += 1
        except AssertionError:
            print(f"Test case {i} FAILED.")
            num_failed += 1
    print(f"{len(simulated_eps_info)} tests total, {num_passed} tests passed, {num_failed} test failed. Pass rate: {num_passed / len(simulated_eps_info) * 100}%.\n")


def main():
    """
    simulated_eps_info: A list of tuples simulateing episodic infomation. 
    Each tuple satisfies the format (task_id, episode_return, episode_length).
    """
    simulated_eps_info = [(0, 5.0, 48), (1, 1.0, 75), (2, 2.0, 36), (2, 4.0, 65), (0, 1.0, 54), (1, 3.0, 82),
                          (0, 2.0, 39), (2, 3.0, 80), (0, 4.0, 57), (1, 0.0, 94), (1, 2.0, 64), (0, 0.0, 45),
                          (2, 1.0, 86), (0, 2.0, 68), (1, 2.0, 92), (2, 1.0, 71), (0, 3.0, 32), (2, 1.0, 47)]
    task_space = DiscreteTaskSpace(3)

    """
    Testing StatRecorder by calculating running average.
    """
    print("Testing StatRecorder by calculating running average: ")
    expected_episode_reward_mean_by_task = [5.0, 1.0, 2.0, 3.0, 3.0, 2.0,
                                            2.6666667, 3.0, 3.0, 1.3333333, 1.5, 2.4,
                                            2.5, 2.3333333, 1.6, 2.2, 2.4285714, 2.0
                                            ]
    expected_episode_reward_std_by_task = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0,
                                           1.6996732, 0.8164966, 1.5811388, 1.2472191, 1.118034, 1.8547237,
                                           1.118034, 1.6996732, 1.0198039, 1.1661904, 1.5907898, 1.1547005]
    expected_episode_length_mean_by_task = [48, 75, 36, 50.5, 51, 78.5,
                                            47, 60.3333333, 49.5, 83.6666667, 78.75, 48.6,
                                            66.75, 51.8333333, 81.4, 67.6, 49, 64.1666667]
    expected_episode_length_std_by_task = [0.0, 0.0, 0.0, 14.5, 3.0, 3.5,
                                           6.164414, 18.2635034, 6.8738635, 7.8457349, 10.8943793, 6.406247, 19.3309984, 9.2990442, 11.0923397, 17.3735431, 11.0582871, 17.620222]
    expected_normalized_reward = [deque([0.0]), deque([0.0]), deque([0.0]), deque([-1.0, 1.0]), deque([1.0, -1.0]), deque([-1.0, 1.0]),
                                  deque([1.3728129, -0.9805807, -0.3922323]), deque([-1.2247449, 1.2247449, 0.0]), deque([1.2649111, -1.2649111, -0.6324555, 0.6324555]), deque(
                                      [-0.2672612, 1.3363062, -1.069045]), deque([-0.4472136, 1.3416408, -1.3416408, 0.4472136]), deque([1.4018261, -0.7548294, -0.2156655, 0.8626622, -1.2939933]),
                                  deque([-0.4472136, 1.3416408, 0.4472136, -1.3416408]), deque([1.5689291, -0.7844645, -0.1961161, 0.9805807, -1.3728129, -0.1961161]), deque([-0.5883484, 1.3728129, -1.5689291, 0.3922323, 0.3922323]), deque(
                                      [-0.1714986, 1.5434873, 0.6859943, -1.0289915, -1.0289915]), deque([1.6164477, -0.8980265, -0.269408, 0.9878292, -1.5266451, -0.269408, 0.3592106]), deque([0.0, 1.7320508, 0.8660254, -0.8660254, -0.8660254, -0.8660254])
                                  ]

    stat_recorder = StatRecorder(task_space)
    test(stat_recorder, simulated_eps_info, expected_episode_reward_mean_by_task, expected_episode_reward_std_by_task,
         expected_episode_length_mean_by_task, expected_episode_length_std_by_task, expected_normalized_reward)

    """
    Testing StatRecorder by calculating based on last n episodes.
    """
    calc_past_n = 3
    print(f"Testing StatRecorder by calculating based on last {calc_past_n} episodes: ")
    expected_episode_reward_mean_by_task = [5.0, 1.0, 2.0, 3.0, 3.0, 2.0,
                                            2.6666667, 3.0, 2.3333333, 1.3333333, 1.6666667, 2.0,
                                            2.6666667, 2.0, 1.3333333, 1.6666667, 1.6666667, 1.0]
    expected_episode_reward_std_by_task = [0.0, 0.0, 0.0, 1.0, 2.0, 1.0,
                                           1.6996732, 0.8164966, 1.2472191, 1.2472191, 1.2472191, 1.6329932,
                                           1.2472191, 1.6329932, 0.942809, 0.942809, 1.2472191, 0.0]
    expected_episode_length_mean_by_task = [48, 75, 36, 50.5, 51, 78.5,
                                            47, 60.3333333, 50, 83.6666667, 80, 47,
                                            77, 56.6666667, 83.3333333, 79, 48.3333333, 68]
    expected_episode_length_std_by_task = [0.0, 0.0, 0.0, 14.5, 3.0, 3.5,
                                           6.164414, 18.2635034, 7.8740079, 7.8457349, 12.328828, 7.4833148,
                                           8.8317609, 9.3926685, 13.6950924, 6.164414, 14.8847424, 16.0623784]
    expected_normalized_reward = [deque([0.0]), deque([0.0]), deque([0.0]),  deque([-1.0, 1.0]), deque([1.0, -1.0]), deque([-1.0, 1.0]),
                                  deque([1.3728129, -0.9805807, -0.3922323]), deque([-1.2247449, 1.2247449, 0.0]), deque([2.1380899, -1.069045, -0.2672612, 1.3363062]), deque(
                                      [-0.2672612, 1.3363062, -1.069045]), deque([-0.5345225, 1.069045, -1.3363062, 0.2672612]), deque([1.8371173, -0.6123724, 0.0, 1.2247449, -1.2247449]),
                                  deque([-0.5345225, 1.069045, 0.2672612, -1.3363062]), deque([1.8371173, -0.6123724, 0.0, 1.2247449, -1.2247449, 0.0]),  deque([-0.3535534, 1.767767, -1.4142136, 0.7071068, 0.7071068]), deque(
                                      [0.3535534, 2.4748737, 1.4142136, -0.7071068, -0.7071068]), deque([2.6726124, -0.5345225, 0.2672612, 1.8708287, -1.3363062, 0.2672612, 1.069045]), deque([100.0, 300.0, 200.0, 0.0, 0.0, 0.0])
                                  ]

    stat_recorder = StatRecorder(task_space, calc_past_n)
    test(stat_recorder, simulated_eps_info, expected_episode_reward_mean_by_task, expected_episode_reward_std_by_task,
         expected_episode_length_mean_by_task, expected_episode_length_std_by_task, expected_normalized_reward)


if __name__ == "__main__":
    main()

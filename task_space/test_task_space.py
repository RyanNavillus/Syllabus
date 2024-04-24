import gymnasium as gym
from syllabus.task_space import TaskSpace

if __name__ == "__main__":
    # Discrete Tests
    task_space = TaskSpace(gym.spaces.Discrete(3), ["a", "b", "c"])

    assert task_space.encode("a") == 0, f"Expected 0, got {task_space.encode('a')}"
    assert task_space.encode("b") == 1, f"Expected 1, got {task_space.encode('b')}"
    assert task_space.encode("c") == 2, f"Expected 2, got {task_space.encode('c')}"
    assert task_space.encode("d") is None, f"Expected None, got {task_space.encode('d')}"

    assert task_space.decode(0) == "a", f"Expected a, got {task_space.decode(0)}"
    assert task_space.decode(1) == "b", f"Expected b, got {task_space.decode(1)}"
    assert task_space.decode(2) == "c", f"Expected c, got {task_space.decode(2)}"
    assert task_space.decode(3) is None, f"Expected None, got {task_space.decode(3)}"
    print("Discrete tests passed!")

    # MultiDiscrete Tests
    task_space = TaskSpace(gym.spaces.MultiDiscrete([3, 2]), [("a", "b", "c"), (1, 0)])

    assert task_space.encode(('a', 1)) == 0, f"Expected 0, got {task_space.encode(('a', 1))}"
    assert task_space.encode(('b', 0)) == 3, f"Expected 3, got {task_space.encode(('b', 0))}"
    assert task_space.encode(('c', 1)) == 4, f"Expected 4, got {task_space.encode(('c', 1))}"

    assert task_space.decode(3) == ('b', 0), f"Expected ('b', 0), got {task_space.decode(3)}"
    assert task_space.decode(5) == ('c', 0), f"Expected ('c', 0), got {task_space.decode(5)}"
    print("MultiDiscrete tests passed!")

    # Box Tests
    task_space = TaskSpace(gym.spaces.Box(low=0, high=1, shape=(2,)), [(0, 0), (0, 1), (1, 0), (1, 1)])

    assert task_space.encode([0.0, 0.0]) == [0.0, 0.0], f"Expected [0.0, 0.0], got {task_space.encode([0.0, 0.0])}"
    assert task_space.encode([0.0, 0.1]) == [0.0, 0.1], f"Expected [0.0, 0.1], got {task_space.encode([0.0, 0.1])}"
    assert task_space.encode([0.1, 0.1]) == [0.1, 0.1], f"Expected [0.1, 0.1], got {task_space.encode([0.1, 0.1])}"
    assert task_space.encode([1.0, 0.1]) == [1.0, 0.1], f"Expected [1.0, 0.1], got {task_space.encode([1.0, 0.1])}"
    assert task_space.encode([1.0, 1.0]) == [1.0, 1.0], f"Expected [1.0, 1.0], got {task_space.encode([1.0, 1.0])}"
    assert task_space.encode([1.2, 1.0]) is None, f"Expected None, got {task_space.encode([1.2, 1.0])}"
    assert task_space.encode([1.0, 1.2]) is None, f"Expected None, got {task_space.encode([1.2, 1.0])}"
    assert task_space.encode([-0.1, 1.0]) is None, f"Expected None, got {task_space.encode([1.2, 1.0])}"

    assert task_space.decode([1.0, 1.0]) == [1.0, 1.0], f"Expected [1.0, 1.0], got {task_space.decode([1.0, 1.0])}"
    assert task_space.decode([0.1, 0.1]) == [0.1, 0.1], f"Expected [0.1, 0.1], got {task_space.decode([0.1, 0.1])}"
    assert task_space.decode([-0.1, 1.0]) is None, f"Expected None, got {task_space.decode([1.2, 1.0])}"
    print("Box tests passed!")

    # Tuple Tests
    task_spaces = (gym.spaces.MultiDiscrete([3, 2]), gym.spaces.Discrete(3))
    task_names = ((("a", "b", "c"), (1, 0)), ("X", "Y", "Z"))
    task_space = TaskSpace(gym.spaces.Tuple(task_spaces), task_names)

    assert task_space.encode((('a', 0), 'Y')) == [1, 1], f"Expected 0, got {task_space.encode((('a', 1),'Y'))}"
    assert task_space.decode([0, 1]) == [('a', 1), 'Y'], f"Expected 0, got {task_space.decode([0, 1])}"
    print("Tuple tests passed!")

    # Dictionary Tests
    task_spaces = gym.spaces.Dict({
        "ext_controller": gym.spaces.MultiDiscrete([5, 2, 2]),
        "inner_state": gym.spaces.Dict(
            {
                "charge": gym.spaces.Discrete(10),
                "system_checks": gym.spaces.Tuple((gym.spaces.MultiDiscrete([3, 2]), gym.spaces.Discrete(3))),
                "job_status": gym.spaces.Dict(
                    {
                        "task": gym.spaces.Discrete(5),
                        "progress": gym.spaces.Box(low=0, high=1, shape=(2,)),
                    }
                ),
            }
        ),
    })
    task_names = {
        "ext_controller": [("a", "b", "c", "d", "e"), (1, 0), ("X", "Y")],
        "inner_state": {
            "charge": [0, 1, 13, 3, 94, 35, 6, 37, 8, 9],
            "system_checks": ((("a", "b", "c"), (1, 0)), ("X", "Y", "Z")),
            "job_status": {
                "task": ["A", "B", "C", "D", "E"],
                "progress": [(0, 0), (0, 1), (1, 0), (1, 1)],
            }
        }
    }
    task_space = TaskSpace(task_spaces, task_names)

    test_val = {
        "ext_controller": ('b', 1, 'X'),
        'inner_state': {
            'charge': 1,
            'system_checks': [('a', 0), 'Y'],
            'job_status': {'task': 'C', 'progress': [0.0, 0.0]}
        }
    }
    decode_val = {
        "ext_controller": 4,
        "inner_state": {
            "charge": 1,
            "system_checks": [1, 1],
            "job_status": {"progress": [0.0, 0.0], "task": 2},
        },
    }

    assert task_space.encode(test_val) == decode_val, f"Expected {decode_val}, \n but got {task_space.encode(test_val)}"
    assert task_space.decode(decode_val) == test_val, f"Expected {test_val}, \n but got {task_space.decode(decode_val)}"

    test_val_2 = {
        "ext_controller": ("e", 1, "Y"),
        "inner_state": {
            "charge": 37,
            "system_checks": [("b", 0), "Z"],
            "job_status": {"progress": [0.0, 0.1], "task": "D"},
        },
    }
    decode_val_2 = {
        "ext_controller": 17,
        "inner_state": {
            "charge": 7,
            "system_checks": [3, 2],
            "job_status": {"progress": [0.0, 0.1], "task": 3},
        },
    }

    assert task_space.encode(test_val_2) == decode_val_2, f"Expected {decode_val_2}, \n but got {task_space.encode(test_val_2)}"
    assert task_space.decode(decode_val_2) == test_val_2, f"Expected {test_val_2}, \n but got {task_space.decode(decode_val_2)}"

    test_val_3 = {
        "ext_controller": ("e", 1, "X"),
        "inner_state": {
            "charge": 8,
            "system_checks": [("c", 0), "X"],
            "job_status": {"progress": [0.5, 0.1], "task": "E"},
        },
    }
    decode_val_3 = {
        "ext_controller": 16,
        "inner_state": {
            "charge": 8,
            "system_checks": [5, 0],
            "job_status": {"progress": [0.5, 0.1], "task": 4},
        },
    }

    assert task_space.encode(test_val_3) == decode_val_3, f"Expected {decode_val_3}, \n but got {task_space.encode(test_val_3)}"
    assert task_space.decode(decode_val_3) == test_val_3, f"Expected {test_val_3}, \n but got {task_space.decode(decode_val_3)}"

    print("Dictionary tests passed!")

    # Test syntactic sugar
    task_space = TaskSpace(3)
    assert task_space.encode(0) == 0, f"Expected 0, got {task_space.encode(0)}"
    assert task_space.encode(1) == 1, f"Expected 1, got {task_space.encode(1)}"
    assert task_space.encode(2) == 2, f"Expected 2, got {task_space.encode(2)}"
    assert task_space.encode(3) is None, f"Expected None, got {task_space.encode(3)}"

    task_space = TaskSpace((2, 4))
    assert task_space.encode((0, 0)) == 0, f"Expected 0, got {task_space.encode((0, 0))}"
    assert task_space.encode((0, 1)) == 1, f"Expected 1, got {task_space.encode((0, 1))}"
    assert task_space.encode((1, 0)) == 4, f"Expected 2, got {task_space.encode((1, 0))}"
    assert task_space.encode((3, 3)) is None, f"Expected None, got {task_space.encode((3, 3))}"

    task_space = TaskSpace((2, 4))
    assert task_space.encode((0, 0)) == 0, f"Expected 0, got {task_space.encode((0, 0))}"
    assert task_space.encode((0, 1)) == 1, f"Expected 1, got {task_space.encode((0, 1))}"
    assert task_space.encode((1, 0)) == 4, f"Expected 2, got {task_space.encode((1, 0))}"
    assert task_space.encode((3, 3)) is None, f"Expected None, got {task_space.encode((3, 3))}"

    task_space = TaskSpace({"map": 5, "level": (4, 10), "difficulty": 3})

    encoding = task_space.encode({"map": 0, "level": (0, 0), "difficulty": 0})
    expected = {"map": 0, "level": 0, "difficulty": 0}

    encoding = task_space.encode({"map": 4, "level": (3, 9), "difficulty": 2})
    expected = {"map": 4, "level": 39, "difficulty": 2}
    assert encoding == expected, f"Expected {expected}, got {encoding}"

    encoding = task_space.encode({"map": 2, "level": (2, 0), "difficulty": 1})
    expected = {"map": 2, "level": 20, "difficulty": 1}
    assert encoding == expected, f"Expected {expected}, got {encoding}"

    encoding = task_space.encode({"map": 5, "level": (2, 11), "difficulty": -1})
    expected = {"map": None, "level": None, "difficulty": None}
    assert encoding == expected, f"Expected {expected}, got {encoding}"
    print("All tests passed!")

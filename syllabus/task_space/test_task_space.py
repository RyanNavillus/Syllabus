import gymnasium as gym
from syllabus.task_space import TaskSpace, FlatTaskSpace, DiscreteTaskSpace, MultiDiscreteTaskSpace, TupleTaskSpace
from syllabus.utils import UsageError


def test_error(space, input_val, error, method="encode"):
    try:
        if method == "encode":
            val = space.encode(input_val)
        else:
            val = space.decode(input_val)
        raise AssertionError(error.format(val))
    except UsageError:
        pass


def test_encoder(space, input_val, expected_output):
    encoded_val = space.encode(input_val)
    assert encoded_val == expected_output, f"Expected {expected_output}, got {encoded_val}"


def test_decoder(space, input_val, expected_output):
    decoded_val = space.decode(input_val)
    assert decoded_val == expected_output, f"Expected {expected_output}, got {decoded_val}"


def test_all_elements(space):
    for element in space.tasks:
        try:
            encoded_val = space.encode(element)
        except Exception as e:
            raise AssertionError(f"Error in encoding {element}") from e
        try:
            decoded_val = space.decode(encoded_val)
        except Exception as e:
            raise AssertionError(f"Error in decoding {encoded_val}") from e
        assert decoded_val == element, f"Expected {element}, got {decoded_val}"


if __name__ == "__main__":
    # # Discrete Tests
    # task_space = TaskSpace(gym.spaces.Discrete(3), ["a", "b", "c"])
    # test_encoder(task_space, "a", 0)
    # test_encoder(task_space, "b", 1)
    # test_encoder(task_space, "c", 2)
    # test_error(task_space, "d", "Expected UsageError, got {}", method="encode")
    # test_decoder(task_space, 0, "a")
    # test_decoder(task_space, 1, "b")
    # test_decoder(task_space, 2, "c")
    # test_error(task_space, 3, "Expected UsageError, got {}", method="decode")
    # test_all_elements(task_space)
    # print("Discrete tests passed!")

    # # MultiDiscrete Tests
    # task_space = TaskSpace(gym.spaces.MultiDiscrete([3, 2]), [("a", "b", "c"), (1, 0)])
    # test_encoder(task_space, ('a', 1), (0, 0))
    # test_encoder(task_space, ('b', 0), (1, 1))
    # test_encoder(task_space, ('c', 1), (2, 0))
    # test_decoder(task_space, (1, 1), ('b', 0))
    # test_decoder(task_space, (2, 1), ('c', 0))
    # test_all_elements(task_space)
    # print("MultiDiscrete tests passed!")

    # # Box Tests
    # task_space = TaskSpace(gym.spaces.Box(low=0, high=1, shape=(2,)), [(0, 0), (0, 1), (1, 0), (1, 1)])

    # assert task_space.encode([0.0, 0.0]) == [0.0, 0.0], f"Expected [0.0, 0.0], got {task_space.encode([0.0, 0.0])}"
    # assert task_space.encode([0.0, 0.1]) == [0.0, 0.1], f"Expected [0.0, 0.1], got {task_space.encode([0.0, 0.1])}"
    # assert task_space.encode([0.1, 0.1]) == [0.1, 0.1], f"Expected [0.1, 0.1], got {task_space.encode([0.1, 0.1])}"
    # assert task_space.encode([1.0, 0.1]) == [1.0, 0.1], f"Expected [1.0, 0.1], got {task_space.encode([1.0, 0.1])}"
    # assert task_space.encode([1.0, 1.0]) == [1.0, 1.0], f"Expected [1.0, 1.0], got {task_space.encode([1.0, 1.0])}"
    # test_error(task_space, [1.2, 1.0], "Expected UsageError, got {}")
    # test_error(task_space, [-0.1, 1.0], "Expected UsageError, got {}")
    # test_error(task_space, [1.0, 1.2], "Expected UsageError, got {}")

    # assert task_space.decode([1.0, 1.0]) == [1.0, 1.0], f"Expected [1.0, 1.0], got {task_space.decode([1.0, 1.0])}"
    # assert task_space.decode([0.1, 0.1]) == [0.1, 0.1], f"Expected [0.1, 0.1], got {task_space.decode([0.1, 0.1])}"
    # test_error(task_space, [-0.1, 1.0], "Expected UsageError, got {}", method="decode")

    # print("Box tests passed!")

    # Tuple Tests
    # task_spaces = (gym.spaces.MultiDiscrete([3, 2]), gym.spaces.Discrete(3))
    # task_names = ((("a", "b", "c"), (1, 0)), ("X", "Y", "Z"))
    # task_space = TaskSpace(gym.spaces.Tuple(task_spaces), task_names)

    # assert task_space.encode((('a', 0), 'Y')) == [1, 1], f"Expected 0, got {task_space.encode((('a', 1),'Y'))}"
    # assert task_space.decode([0, 1]) == [('a', 1), 'Y'], f"Expected 0, got {task_space.decode([0, 1])}"
    # print("Tuple tests passed!")

    # Dictionary Tests
    # task_spaces = gym.spaces.Dict({
    #     "ext_controller": gym.spaces.MultiDiscrete([5, 2, 2]),
    #     "inner_state": gym.spaces.Dict(
    #         {
    #             "charge": gym.spaces.Discrete(10),
    #             "system_checks": gym.spaces.Tuple((gym.spaces.MultiDiscrete([3, 2]), gym.spaces.Discrete(3))),
    #             "job_status": gym.spaces.Dict(
    #                 {
    #                     "task": gym.spaces.Discrete(5),
    #                     "progress": gym.spaces.Box(low=0, high=1, shape=(2,)),
    #                 }
    #             ),
    #         }
    #     ),
    # })
    # task_names = {
    #     "ext_controller": [("a", "b", "c", "d", "e"), (1, 0), ("X", "Y")],
    #     "inner_state": {
    #         "charge": [0, 1, 13, 3, 94, 35, 6, 37, 8, 9],
    #         "system_checks": ((("a", "b", "c"), (1, 0)), ("X", "Y", "Z")),
    #         "job_status": {
    #             "task": ["A", "B", "C", "D", "E"],
    #             "progress": [(0, 0), (0, 1), (1, 0), (1, 1)],
    #         }
    #     }
    # }
    # task_space = TaskSpace(task_spaces, task_names)

    # test_val = {
    #     "ext_controller": ('b', 1, 'X'),
    #     'inner_state': {
    #         'charge': 1,
    #         'system_checks': [('a', 0), 'Y'],
    #         'job_status': {'task': 'C', 'progress': [0.0, 0.0]}
    #     }
    # }
    # decode_val = {
    #     "ext_controller": 4,
    #     "inner_state": {
    #         "charge": 1,
    #         "system_checks": [1, 1],
    #         "job_status": {"progress": [0.0, 0.0], "task": 2},
    #     },
    # }

    # assert task_space.encode(test_val) == decode_val, f"Expected {decode_val}, \n but got {task_space.encode(test_val)}"
    # assert task_space.decode(decode_val) == test_val, f"Expected {test_val}, \n but got {task_space.decode(decode_val)}"

    # test_val_2 = {
    #     "ext_controller": ("e", 1, "Y"),
    #     "inner_state": {
    #         "charge": 37,
    #         "system_checks": [("b", 0), "Z"],
    #         "job_status": {"progress": [0.0, 0.1], "task": "D"},
    #     },
    # }
    # decode_val_2 = {
    #     "ext_controller": 17,
    #     "inner_state": {
    #         "charge": 7,
    #         "system_checks": [3, 2],
    #         "job_status": {"progress": [0.0, 0.1], "task": 3},
    #     },
    # }

    # assert task_space.encode(test_val_2) == decode_val_2, f"Expected {decode_val_2}, \n but got {task_space.encode(test_val_2)}"
    # assert task_space.decode(decode_val_2) == test_val_2, f"Expected {test_val_2}, \n but got {task_space.decode(decode_val_2)}"

    # test_val_3 = {
    #     "ext_controller": ("e", 1, "X"),
    #     "inner_state": {
    #         "charge": 8,
    #         "system_checks": [("c", 0), "X"],
    #         "job_status": {"progress": [0.5, 0.1], "task": "E"},
    #     },
    # }
    # decode_val_3 = {
    #     "ext_controller": 16,
    #     "inner_state": {
    #         "charge": 8,
    #         "system_checks": [5, 0],
    #         "job_status": {"progress": [0.5, 0.1], "task": 4},
    #     },
    # }

    # assert task_space.encode(test_val_3) == decode_val_3, f"Expected {decode_val_3}, \n but got {task_space.encode(test_val_3)}"
    # assert task_space.decode(decode_val_3) == test_val_3, f"Expected {test_val_3}, \n but got {task_space.decode(decode_val_3)}"

    # print("Dictionary tests passed!")

    # # Test syntactic sugar
    # task_space = TaskSpace(3)
    # assert task_space.encode(0) == 0, f"Expected 0, got {task_space.encode(0)}"
    # assert task_space.encode(1) == 1, f"Expected 1, got {task_space.encode(1)}"
    # assert task_space.encode(2) == 2, f"Expected 2, got {task_space.encode(2)}"
    # test_error(task_space, 3, "Expected UsageError, got {}", method="encode")

    # task_space = TaskSpace((3, 2))

    # task_space = TaskSpace((2, 4))
    # assert task_space.encode((0, 0)) == (0, 0), f"Expected 0, got {task_space.encode((0, 0))}"
    # assert task_space.encode((0, 1)) == (0, 1), f"Expected 1, got {task_space.encode((0, 1))}"
    # assert task_space.encode((1, 0)) == (1, 0), f"Expected 2, got {task_space.encode((1, 0))}"
    # test_error(task_space, (3, 3), "Expected UsageError, got {}", method="encode")

    # task_space = TaskSpace({"map": 5, "level": (4, 10), "difficulty": 3})

    # encoding = task_space.encode({"map": 0, "level": (0, 0), "difficulty": 0})
    # expected = {"map": 0, "level": 0, "difficulty": 0}

    # encoding = task_space.encode({"map": 4, "level": (3, 9), "difficulty": 2})
    # expected = {"map": 4, "level": 39, "difficulty": 2}
    # assert encoding == expected, f"Expected {expected}, got {encoding}"

    # encoding = task_space.encode({"map": 2, "level": (2, 0), "difficulty": 1})
    # expected = {"map": 2, "level": 20, "difficulty": 1}
    # assert encoding == expected, f"Expected {expected}, got {encoding}"

    # encoding = task_space.encode({"map": 5, "level": (2, 11), "difficulty": -1})
    # expected = {"map": None, "level": None, "difficulty": None}
    # assert encoding == expected, f"Expected {expected}, got {encoding}"
    # print("All tests passed!")

    # Discrete Tests
    task_space = DiscreteTaskSpace(gym.spaces.Discrete(3), ["a", "b", "c"])

    test_encoder(task_space, "a", 0)
    test_encoder(task_space, "b", 1)
    test_encoder(task_space, "c", 2)
    test_error(task_space, "d", "Expected UsageError, got {}", method="encode")

    test_decoder(task_space, 0, "a")
    test_decoder(task_space, 1, "b")
    test_decoder(task_space, 2, "c")
    test_error(task_space, 3, "Expected UsageError, got {}", method="decode")
    test_all_elements(task_space)

    task_space = DiscreteTaskSpace(4)
    test_encoder(task_space, 0, 0)
    test_encoder(task_space, 1, 1)
    test_encoder(task_space, 2, 2)
    test_error(task_space, 4, "Expected UsageError, got {}", method="encode")

    test_decoder(task_space, 0, 0)
    test_decoder(task_space, 1, 1)
    test_decoder(task_space, 2, 2)
    test_error(task_space, 4, "Expected UsageError, got {}", method="decode")
    test_all_elements(task_space)
    print("Discrete tests passed!")

    # MultiDiscrete Tests
    task_space = MultiDiscreteTaskSpace(([3, 2]), [("a", "b", "c"), (1, 0)], flatten=False)
    test_encoder(task_space, ('a', 1), (0, 0))
    test_encoder(task_space, ('b', 0), (1, 1))
    test_encoder(task_space, ('c', 1), (2, 0))
    test_decoder(task_space, (1, 1), ('b', 0))
    test_decoder(task_space, (2, 1), ('c', 0))
    test_all_elements(task_space)

    task_space = MultiDiscreteTaskSpace(([3, 2]), [("a", "b", "c"), (1, 0)], flatten=True)
    test_encoder(task_space, ('a', 1), 0)
    test_encoder(task_space, ('b', 0), 3)
    test_encoder(task_space, ('c', 1), 4)
    test_decoder(task_space, 3, ('b', 0))
    test_decoder(task_space, 5, ('c', 0))
    test_all_elements(task_space)
    print("MultiDiscrete tests passed!")

    # Tuple Tests
    task_spaces = (MultiDiscreteTaskSpace([3, 2], (("a", "b", "c"), (1, 0))), DiscreteTaskSpace(3, ("X", "Y", "Z")))
    task_space = TupleTaskSpace(task_spaces, space_names=None, flatten=True)
    test_encoder(task_space, (('a', 0), 'Y'), 4)
    test_decoder(task_space, 1, (('a', 1), 'Y'))
    test_error(task_space, (('d', 0), 'Y'), "Expected UsageError, got {}", method="encode")
    test_error(task_space, (('a', 0), 'Y', 0), "Expected UsageError, got {}", method="encode")
    test_error(task_space, 18, "Expected UsageError, got {}", method="decode")
    test_error(task_space, -1, "Expected UsageError, got {}", method="decode")
    test_all_elements(task_space)
    print("Tuple tests passed!")

    # Dictionary Tests
    # task_spaces = gym.spaces.Dict({
    #     "ext_controller": gym.spaces.MultiDiscrete([3, 2]),
    #     "inner_state": gym.spaces.Dict({"charge": gym.spaces.Discrete(4)}),
    # })
    # task_names = {
    #     "ext_controller": [("a", "b", "c"), ("X", "Y")],
    #     "inner_state": {"charge": [0, 1, 13, 3]},
    # }
    # task_space = FlatTaskSpace(task_spaces, task_names)

    # test_val = {
    #     "ext_controller": ('b', 'X'),
    #     'inner_state': {'charge': 1},
    # }
    # decode_val = 10

    # assert task_space.encode(test_val) == decode_val, f"Expected {decode_val}, \n but got {task_space.encode(test_val)}"
    # assert task_space.decode(decode_val) == test_val, f"Expected {test_val}, \n but got {task_space.decode(decode_val)}"

    # test_val_2 = {
    #     "ext_controller": ("e", 1, "Y"),
    #     "inner_state": {
    #         "charge": 37,
    #         "system_checks": [("b", 0), "Z"],
    #         "job_status": {"progress": [0.0, 0.1], "task": "D"},
    #     },
    # }
    # decode_val_2 = {
    #     "ext_controller": 17,
    #     "inner_state": {
    #         "charge": 7,
    #         "system_checks": [3, 2],
    #         "job_status": {"progress": [0.0, 0.1], "task": 3},
    #     },
    # }

    # assert task_space.encode(
    #     test_val_2) == decode_val_2, f"Expected {decode_val_2}, \n but got {task_space.encode(test_val_2)}"
    # assert task_space.decode(
    #     decode_val_2) == test_val_2, f"Expected {test_val_2}, \n but got {task_space.decode(decode_val_2)}"

    # test_val_3 = {
    #     "ext_controller": ("e", 1, "X"),
    #     "inner_state": {
    #         "charge": 8,
    #         "system_checks": [("c", 0), "X"],
    #         "job_status": {"progress": [0.5, 0.1], "task": "E"},
    #     },
    # }
    # decode_val_3 = {
    #     "ext_controller": 16,
    #     "inner_state": {
    #         "charge": 8,
    #         "system_checks": [5, 0],
    #         "job_status": {"progress": [0.5, 0.1], "task": 4},
    #     },
    # }

    # assert task_space.encode(
    #     test_val_3) == decode_val_3, f"Expected {decode_val_3}, \n but got {task_space.encode(test_val_3)}"
    # assert task_space.decode(
    #     decode_val_3) == test_val_3, f"Expected {test_val_3}, \n but got {task_space.decode(decode_val_3)}"

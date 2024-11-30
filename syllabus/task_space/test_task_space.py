import gymnasium as gym
import numpy as np

from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace, TupleTaskSpace, BoxTaskSpace
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


def test_random_elements(space):
    for _ in range(100):
        element = space.sample()
        try:
            encoded_val = space.encode(element)
        except Exception as e:
            raise AssertionError(f"Error in encoding {element}") from e
        try:
            decoded_val = space.decode(encoded_val)
        except Exception as e:
            raise AssertionError(f"Error in decoding {encoded_val}") from e
        assert np.allclose(decoded_val, element), f"Expected {element}, got {decoded_val}"


if __name__ == "__main__":
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

    # Box Tests
    task_space = BoxTaskSpace(gym.spaces.Box(low=0, high=1, shape=(2, 2)))
    test_encoder(task_space, [[0, 1], [1, 0]], [[0, 1], [1, 0]])
    test_encoder(task_space, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    test_encoder(task_space, [[0.5, 0.2], [0.1, 0.9]], [[0.5, 0.2], [0.1, 0.9]])
    test_decoder(task_space, [[0, 1], [1, 0]], [[0, 1], [1, 0]])
    test_decoder(task_space, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    test_decoder(task_space, [[0.5, 0.2], [0.1, 0.9]], [[0.5, 0.2], [0.1, 0.9]])
    test_random_elements(task_space)
    print("Box tests passed!")

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

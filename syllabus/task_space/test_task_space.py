import gymnasium as gym
from syllabus.task_space import TaskSpace

if __name__ == "__main__":
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

    task_space = TaskSpace(gym.spaces.MultiDiscrete([3]), [("a", "b", "c")])
    assert task_space.encode(("a",)) == 0, f"Expected 0, got {task_space.encode(('a',))}"
    assert task_space.encode(("b",)) == 1, f"Expected 1, got {task_space.encode(('b',))}"
    assert task_space.encode(("c",)) == 2, f"Expected 2, got {task_space.encode(('c',))}"
    assert task_space.encode(("d",)) is None, f"Expected None, got {task_space.encode(('d',))}"

    assert task_space.decode(0) == ("a",), f"Expected a, got {task_space.decode(0)}"
    assert task_space.decode(1) == ("b",), f"Expected b, got {task_space.decode(1)}"
    assert task_space.decode(2) == ("c",), f"Expected c, got {task_space.decode(2)}"
    assert task_space.decode(3) is None, f"Expected None, got {task_space.decode(3)}"

    task_space = TaskSpace(gym.spaces.MultiDiscrete([1, 1, 1]), [("a",), ("b",), ("c",)])
    assert task_space.encode(("a", "b", "c")) == 0, f"Expected 0, got {task_space.encode(('a', 'b', 'c'))}"
    assert task_space.encode(("a", "b", "d")) is None, f"Expected None, got {task_space.encode(('a', 'b', 'd'))}"

    assert task_space.decode(0) == ("a", "b", "c"), f"Expected a, got {task_space.decode(0)}"
    assert task_space.decode(1) is None, f"Expected None, got {task_space.decode(1)}"

    task_space = TaskSpace(gym.spaces.MultiDiscrete([3, 2]), [("a", "b", "c"), (1, 2)])
    assert task_space.encode(("a", 1)) == 0, f"Expected 0, got {task_space.encode(('a', 1))}"
    assert task_space.encode(("b", 2)) == 3, f"Expected 1, got {task_space.encode(('b', 2))}"
    assert task_space.encode(("c", 2)) == 5, f"Expected 2, got {task_space.encode(('c', 2))}"
    assert task_space.encode(("d", 0)) is None, f"Expected None, got {task_space.encode(('d', 0))}"

    assert task_space.decode(0) == ("a", 1), f"Expected a, got {task_space.decode(0)}"
    assert task_space.decode(1) == ("a", 2), f"Expected b, got {task_space.decode(1)}"
    assert task_space.decode(2) == ("b", 1), f"Expected c, got {task_space.decode(2)}"
    assert task_space.decode(6) is None, f"Expected None, got {task_space.decode(3)}"
    print("MultiDiscrete tests passed!")

    # Test syntactic sugar
    task_space = TaskSpace(3)
    assert task_space.encode(0) == 0, f"Expected 0, got {task_space.encode(0)}"
    assert task_space.encode(1) == 1, f"Expected 1, got {task_space.encode(1)}"
    assert task_space.encode(2) == 2, f"Expected 2, got {task_space.encode(2)}"
    assert task_space.encode(3) is None, f"Expected None, got {task_space.encode(3)}"

    task_space = TaskSpace((3, 2))

    print("All tests passed!")

import gymnasium as gym
from task_space import TaskSpace

if __name__ == "__main__":
    task_space = TaskSpace(gym.spaces.Discrete(3), ["a", "b", "c"])
    assert task_space.encode("a") == 0, f"Expected 0, got {task_space.encode('a')}"
    assert task_space.encode("b") == 1, f"Expected 1, got {task_space.encode('b')}"
    assert task_space.encode("c") == 2, f"Expected 2, got {task_space.encode('c')}"
    assert task_space.encode("d") == None, f"Expected None, got {task_space.encode('d')}"

    assert task_space.decode(0) == "a", f"Expected a, got {task_space.decode(0)}"
    assert task_space.decode(1) == "b", f"Expected b, got {task_space.decode(1)}"
    assert task_space.decode(2) == "c", f"Expected c, got {task_space.decode(2)}"
    assert task_space.decode(3) == None, f"Expected None, got {task_space.decode(3)}"
    print("Discrete tests passed!")

    task_space = TaskSpace(gym.spaces.MultiDiscrete([3,2]), [("a", "b", "c"),(1,0)])
    # [('a', '1'), ('a', '0'), ('b', '1'), ('b', '0'), ('c', '1'), ('c', '0')]
    assert task_space.encode(('a', 1)) == 0, f"Expected 0, got {task_space.encode(('a', 1))}"
    assert task_space.encode(('b', 0)) == 3, f"Expected 3, got {task_space.encode(('b', 0))}"
    assert task_space.encode(('c', 1)) == 4, f"Expected 4, got {task_space.encode(('c', 1))}"

    assert task_space.decode(3) == ('b', 0), f"Expected ('b', 0), got {task_space.decode(3)}"
    assert task_space.decode(5) == ('c', 0), f"Expected ('c', 0), got {task_space.decode(5)}"
    print("MultiDiscrete tests passed!")

    task_space = TaskSpace(gym.spaces.Box(low=0, high=1, shape=(2,)), [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert task_space.encode([0.0, 0.0]) == [0.0, 0.0], f"Expected [0.0, 0.0], got {task_space.encode([0.0, 0.0])}"
    assert task_space.encode([0.0, 0.1]) == [0.0, 0.1], f"Expected [0.0, 0.1], got {task_space.encode([0.0, 0.1])}"
    assert task_space.encode([0.1, 0.1]) == [0.1, 0.1], f"Expected [0.1, 0.1], got {task_space.encode([0.1, 0.1])}"
    assert task_space.encode([1.0, 0.1]) == [1.0, 0.1], f"Expected [1.0, 0.1], got {task_space.encode([1.0, 0.1])}"
    assert task_space.encode([1.0, 1.0]) == [1.0, 1.0], f"Expected [1.0, 1.0], got {task_space.encode([1.0, 1.0])}"
    assert task_space.encode([1.2, 1.0]) == None, f"Expected None, got {task_space.encode([1.2, 1.0])}"
    assert task_space.encode([1.0, 1.2]) == None, f"Expected None, got {task_space.encode([1.2, 1.0])}"
    assert task_space.encode([-0.1, 1.0]) == None, f"Expected None, got {task_space.encode([1.2, 1.0])}"

    assert task_space.decode([1.0, 1.0]) == [1.0, 1.0], f"Expected [1.0, 1.0], got {task_space.decode([1.0, 1.0])}"
    assert task_space.decode([0.1, 0.1]) == [0.1, 0.1], f"Expected [0.1, 0.1], got {task_space.decode([0.1, 0.1])}"
    assert task_space.decode([-0.1, 1.0]) == None, f"Expected None, got {task_space.decode([1.2, 1.0])}"
    print("Box tests passed!")

    task_space = TaskSpace(gym.spaces.Tuple((gym.spaces.MultiDiscrete([3,2]),gym.spaces.Discrete(3))), ((("a", "b", "c"),(1,0)),("X","Y","Z")))
    assert task_space.encode((('a', 1),'Y')) == [0, 1], f"Expected 0, got {task_space.encode((('a', 1),'Y'))}"
    assert task_space.decode([0,1]) == [('a', 1), 'Y'], f"Expected 0, got {task_space.decode([0,1])}"
    print("Tuple tests passed!")

    di = gym.spaces.Dict(  
    {
        "ext_controller": gym.spaces.MultiDiscrete([5, 2, 2]),
        "inner_state": gym.spaces.Dict(
            {
                "charge": gym.spaces.Discrete(10),
                "system_checks": gym.spaces.Tuple((gym.spaces.MultiDiscrete([3,2]),gym.spaces.Discrete(3))),
                "job_status": gym.spaces.Dict(
                    {
                        "task": gym.spaces.Discrete(5),
                        "progress": gym.spaces.Box(low=0, high=1, shape=(2,0)),
                    }
                ),
            }
        ),
    }
    )
    
    example =  {
        "ext_controller": [("a", "b", "c"),(1,0),("X","Y")],
        "inner_state": {
                "charge": [0,1,2,3,4,5,6,7,8,9],
                "system_checks": ((("a", "b", "c"),(1,0)),("X","Y","Z")),
                "job_status": {
                        "task": ["A","B", "C", "D", "E"],
                        "progress": [(0, 0), (0, 1), (1, 0), (1, 1)],
                    }
            
            }
    }
    taskspace = TaskSpace(di, example)


    a = {
            'ext_controller' : ["b", 1, "X"],
            'inner_state' : {
                'charge' : 1,
                'system_checks' : (["a", 0], "Y"),
                'job_status' : {
                    'progress' : [1.0, 0.1],
                    'task' : "C",

                }

            }
       }

    
    assert task_space.encode({
            'ext_controller' : ["b", 1, "X"],
            'inner_state' : {
                'charge' : 1,
                'system_checks' : (["a", 0], "Y"),
                'job_status' : {
                    'progress' : [1.0, 0.1],
                    'task' : "C",

                }

            }
       }) == [0, 1], f"Expected 0, got {task_space.encode(a)}"

    # OrderedDict(
    #     [
    #         (
    #             'ext_controller', array([2, 0, 1])), 
    #             ('inner_state', OrderedDict(
    #                 [
    #                     ('charge', 14), 
    #                     ('job_status', OrderedDict(
    #                         [
    #                             ('progress', array(15.613569, dtype=float32)), 
    #                             ('task', 4)
    #                             ]
    #                             )
    #                             ), 
    #     ('system_checks', (array([0, 0]), 2))
    #     ]
    #     )
    #     )
    #     ]
    #     )

       
    #             'ext_controller', array([2, 0, 1])), 
    #             ('inner_state', OrderedDict(
    #                 [
    #                     ('charge', 14), 
    #                     ('job_status', OrderedDict(
    #                         [
    #                             ('progress', array(15.613569, dtype=float32)), 
    #                             ('task', 4)
    #                             ]
    #                             )
    #                             ), 
    #     ('system_checks', (array([0, 0]), 2))
    #     ]
    #     )
    #     )
    #     ]
    #     )


    # Test syntactic sugar
    task_space = TaskSpace(3)
    assert task_space.encode(0) == 0, f"Expected 0, got {task_space.encode(0)}"
    assert task_space.encode(1) == 1, f"Expected 1, got {task_space.encode(1)}"
    assert task_space.encode(2) == 2, f"Expected 2, got {task_space.encode(2)}"
    assert task_space.encode(3) is None, f"Expected None, got {task_space.encode(3)}"

    print("All tests passed!")

import warnings

from .cartpole_task_wrapper import CartPoleTaskWrapper

try:
    from .minigrid_task_wrapper import MinigridTaskWrapper
except ImportError as e:
    warnings.warn(f"Unable to import the following minigrid dependencies: {e.name}", stacklevel=2)

try:
    from .nethack_wrappers import NethackTaskWrapper, NethackSeedWrapper, NetHackCollect, NetHackDescend, NetHackSatiate, NetHackScoutClipped, NetHackSeed
except ImportError as e:
    warnings.warn(f"Unable to import the following nethack dependencies: {e.name}", stacklevel=2)

try:
    from .procgen_task_wrapper import ProcgenTaskWrapper
except ImportError as e:
    warnings.warn(f"Unable to import the following procgen dependencies: {e.name}", stacklevel=2)

try:
    from .pistonball_task_wrapper import PistonballTaskWrapper
except ImportError as e:
    warnings.warn(f"Unable to import the following pistonball dependencies: {e.name}", stacklevel=2)

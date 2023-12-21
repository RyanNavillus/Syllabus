import warnings

from .cartpole_task_wrapper import CartPoleTaskWrapper

from .minigrid_task_wrapper import MinigridTaskWrapper

try:
    from .minihack_task_wrapper import MinihackTaskWrapper
except ImportError:
    warnings.warn("Unable to import minihack.")
    pass

# from .pistonball_task_wrapper import PistonballTaskWrapper
try:
    from .nethack_task_wrapper import NethackTaskWrapper
except ImportError:
    warnings.warn("Unable to import nle.")


try:
    from .procgen_task_wrapper import ProcgenTaskWrapper
except ImportError:
    warnings.warn("Unable to import procgen.")
    pass

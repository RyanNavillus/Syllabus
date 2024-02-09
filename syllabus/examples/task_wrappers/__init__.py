import warnings

from .cartpole_task_wrapper import CartPoleTaskWrapper

try:
    from .minigrid_task_wrapper import MinigridTaskWrapper
except ImportError  as e:
    warnings.warn(f"Unable to import the following minigrid dependencies: {e.name}")

try:
    from .minihack_task_wrapper import MinihackTaskWrapper
except ImportError  as e:
    warnings.warn(f"Unable to import the following minihack dependencies: {e.name}")

try:
    from .nethack_wrappers import NethackTaskWrapper    #, RenderCharImagesWithNumpyWrapperV2
except ImportError as e:
    warnings.warn(f"Unable to import the following nle dependencies: {e.name}")

try:
    from .procgen_task_wrapper import ProcgenTaskWrapper
except ImportError  as e:
    warnings.warn(f"Unable to import the following procgen dependencies: {e.name}")

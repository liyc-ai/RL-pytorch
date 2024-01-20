__version__ = "1.2.0"

from typing import Any, Dict

from rlplugs.logger import LoggerType

from rlpyt.algo import AGENTS, BaseRLAgent


def make(cfg: Dict[str, Any], logger: LoggerType) -> BaseRLAgent:
    """Create a RL agent
    """
    def _instantiate_agent(cfg: Dict, logger: LoggerType) -> BaseRLAgent:
        """For annotation
        """
        return AGENTS[cfg["agent"]["algo"]](cfg, logger)
    
    agent = _instantiate_agent(cfg, logger)
    agent.setup_model()
    return agent

"""The default MLP agent package."""

from fsrl.agent.base_agent import BaseAgent, OffpolicyAgent, OnpolicyAgent

from fsrl.agent.ppo_lag_agent import PPOLagAgent

from fsrl.agent.trpo_lag_agent import TRPOLagAgent
from fsrl.agent.locu_agent import LOCUAgent
from fsrl.agent.crpo_agent import CRPOAgent
from fsrl.agent.pcrpo_agent import PCRPOAgent
from fsrl.agent.sdac_actor_agent import SDACACTORAgent
__all__ = [
    "BaseAgent", "OffpolicyAgent", "OnpolicyAgent", 
   "PPOLagAgent", "TRPOLagAgent", 
   
    "LOCUAgent", "CRPOAgent", "PCRPOAgent", "SDACACTORAgent"
]

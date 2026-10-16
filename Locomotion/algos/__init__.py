from algos.crpo.agent import Agent as CRPO
from algos.sdac.agent import Agent as SDAC
from algos.locu.agent import Agent as LOCU
from algos.trpol.agent import Agent as TRPOL
algo_dict = {}
algo_dict['crpo'] = CRPO
algo_dict['sdac'] = SDAC
algo_dict['locu'] = LOCU
algo_dict['trpol'] = TRPOL
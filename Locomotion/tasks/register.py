from gymnasium.envs.registration import register

register(
    id='Laikago-v0',
    entry_point='tasks.laikago.env_v0:Env',
)

register(
    id='Cassie-v0',
    entry_point='tasks.cassie.env_v0:Env',
)

register(
    id='CassieGoalHazard-v0',
    entry_point='tasks.cassie_goal.env_v0:Env',
)

register(
    id='CassieGoalRedun-v0',
    entry_point='tasks.cassie_goal_redun.env_v0:Env',
)

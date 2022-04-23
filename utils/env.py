def add_env_info(configs, env=None, env_info=None):
    if env != None:
        configs["state_dim"], configs["action_dim"], configs["action_high"] = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
            env.action_space.high[0],
        )
    elif env_info == None:
        configs = {**configs, **env_info}
    else:
        raise ValueError("env or env_info must be not None")

    return configs

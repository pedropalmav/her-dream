from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def make_env(config, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import her_dream.envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size, seed=config.seed + id)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import her_dream.envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.gray,
            noops=config.noops,
            lives=config.lives,
            sticky=config.sticky,
            actions=config.actions,
            length=config.time_limit,
            pooling=config.pooling,
            aggregate=config.aggregate,
            resize=config.resize,
            autostart=config.autostart,
            clip_reward=config.clip_reward,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from her_dream.envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import her_dream.envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
        # No GoalImageObservation: crafter has no goal cell to render, so
        # goal_sample="image" is unavailable here (buffer/imagination/random are).
        env = wrappers.GoalConditioned(env, config)
    elif suite == "metaworld":
        import her_dream.envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    elif suite in ("goal-grid", "random-goal", "fixed-goal"):
        # `random-goal` and `fixed-goal` were separate suites backed by separate
        # env classes; both are now cookie_env.GoalGrid, selected by goal_pos.
        # Both names stay accepted because every archived run's .hydra/config.yaml
        # carries one of them.
        from cookie_env.envs.goal_grid import make_goal_grid_env

        # A configured goal cell pins the square; its absence (or an explicit
        # null) resamples it every episode. Old random-goal configs simply have
        # no goal_pos_x/y key, so they land on None without needing an edit.
        goal_pos_x = getattr(config, "goal_pos_x", None)
        goal_pos_y = getattr(config, "goal_pos_y", None)
        goal_pos = None if goal_pos_x is None or goal_pos_y is None else (goal_pos_x, goal_pos_y)

        agent_start_pos = (
            None
            if getattr(config, "agent_start_random", False)
            else (config.agent_start_pos_x, config.agent_start_pos_y)
        )
        env = make_goal_grid_env(
            size=config.env_size,
            agent_start_dir=config.agent_start_dir,
            agent_start_pos=agent_start_pos,
            goal_pos=goal_pos,
            max_steps=config.time_limit,
            render_mode=getattr(config, "render_mode", "rgb_array"),
        )
        env = wrappers.MissionGridWrapper(env) if config.mission_text else wrappers.MiniGridWrapper(env)
        env = wrappers.OneHotAction(env)
        env = wrappers.GoalConditioned(env, config)
        # The generator's own goal_pos is overwritten per call by
        # GoalImageGenerator.observation(), so it is pinned here only to keep the
        # auxiliary env deterministic before the first assignment.
        env = wrappers.GoalImageObservation(env, lambda: make_goal_grid_env(size=config.env_size, goal_pos=(1, 1)))
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)

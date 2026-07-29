from . import parallel, wrappers


def make_envs(config):
    def env_constructor(idx):
        return lambda: make_env(config, idx)

    train_envs = parallel.ParallelEnv(env_constructor, config.env_num, config.device)
    eval_envs = parallel.ParallelEnv(env_constructor, config.eval_episode_num, config.device)
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


def _goal_image_kwargs(config):
    """Pose configuration for the goal_sample="image" renderer.

    Read with getattr defaults so configs saved before these keys existed
    (`<logdir>/.hydra/config.yaml`, replayed by evaluate.py and the experiment
    scripts) still build an env.
    """
    return {
        "agent_on_goal": getattr(config, "goal_image_on_goal", False),
        "agent_dir": getattr(config, "goal_image_agent_dir", None),
    }


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
    elif suite == "metaworld":
        import her_dream.envs.metaworld as metaworld

        env = metaworld.MetaWorld(
            task,
            config.action_repeat,
            config.size,
            config.camera,
            config.seed + id,
        )
    elif suite == "random-goal":
        import her_dream.envs.fixed_goal as fixed_goal
        import her_dream.envs.random_goal as random_goal

        agent_start_pos = (
            None
            if getattr(config, "agent_start_random", False)
            else (config.agent_start_pos_x, config.agent_start_pos_y)
        )
        env = random_goal.make_random_goal_env(
            size=config.env_size,
            agent_start_dir=config.agent_start_dir,
            agent_start_pos=agent_start_pos,
            max_steps=config.time_limit,
            render_mode=config.render_mode,
        )
        env = wrappers.MissionGridWrapper(env) if config.mission_text else wrappers.MiniGridWrapper(env)

        env = wrappers.OneHotAction(env)
        env = wrappers.GoalConditioned(env, config)
        # Auxiliary FixedGoal generator honours goal_pos, so it can render the
        # green square at the live (randomly placed) goal position.
        env = wrappers.GoalImageObservation(
            env,
            lambda: fixed_goal.make_fixed_goal_env(size=config.env_size),
            **_goal_image_kwargs(config),
        )

    elif suite == "fixed-goal":
        import her_dream.envs.fixed_goal as fixed_goal

        agent_start_pos = (config.agent_start_pos_x, config.agent_start_pos_y)
        goal_pos = (config.goal_pos_x, config.goal_pos_y)
        env = fixed_goal.make_fixed_goal_env(
            size=config.env_size,
            agent_start_dir=config.agent_start_dir,
            agent_start_pos=agent_start_pos,
            goal_pos=goal_pos,
            max_steps=config.time_limit,
        )
        env = wrappers.MissionGridWrapper(env) if config.mission_text else wrappers.MiniGridWrapper(env)
        env = wrappers.OneHotAction(env)
        env = wrappers.GoalConditioned(env, config)
        env = wrappers.GoalImageObservation(
            env,
            lambda: fixed_goal.make_fixed_goal_env(size=config.env_size),
            **_goal_image_kwargs(config),
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)

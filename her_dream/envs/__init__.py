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
        env = wrappers.GoalImageObservation(env, lambda: fixed_goal.make_fixed_goal_env(size=config.env_size))

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
        env = wrappers.GoalImageObservation(env, lambda: fixed_goal.make_fixed_goal_env(size=config.env_size))

    elif suite == "lava-grid":
        # The first env here that can terminate on something the agent does. It comes
        # from the sibling cookie-env package (`LavaGrid` = its `GoalGrid` plus lava
        # cells that end the episode on contact), which is why this branch imports
        # across the repo boundary instead of adding another module under envs/.
        from cookie_env.envs.lava_grid import make_lava_grid_env

        # Same goal_pos convention as cookie-env and MiniGrid's agent_start_pos: a
        # configured cell pins the green square, null on either axis resamples it.
        goal_pos_x = getattr(config, "goal_pos_x", None)
        goal_pos_y = getattr(config, "goal_pos_y", None)
        goal_pos = None if goal_pos_x is None or goal_pos_y is None else (goal_pos_x, goal_pos_y)
        agent_start_pos = (
            None
            if getattr(config, "agent_start_random", False)
            else (config.agent_start_pos_x, config.agent_start_pos_y)
        )
        # `lava_penalty` is deliberately not forwarded: `trainer._apply_reward`
        # overwrites reward with the goal-conditioned latent reward before the
        # transition reaches the buffer, so nothing the env returns survives. The
        # sign problem that knob exists for is fixed on this side instead, with the
        # top-level `reward_offset` (train.py:validate_config warns when it is 0).
        # `lava_pos` pins the lava instead of resampling it every reset; left null it
        # resamples `n_lava` cells. The analysis scripts under experiments/ set it so
        # they can dictate a whole layout (see experiments/common/envs.py).
        lava_pos = getattr(config, "lava_pos", None)
        env = make_lava_grid_env(
            size=config.env_size,
            agent_start_dir=config.agent_start_dir,
            agent_start_pos=agent_start_pos,
            goal_pos=goal_pos,
            max_steps=config.time_limit,
            n_lava=config.n_lava,
            lava_pos=None if lava_pos is None else [tuple(cell) for cell in lava_pos],
            render_mode=getattr(config, "render_mode", "rgb_array"),
        )
        env = wrappers.MissionGridWrapper(env) if config.mission_text else wrappers.MiniGridWrapper(env)
        env = wrappers.OneHotAction(env)
        env = wrappers.GoalConditioned(env, config)
        # The auxiliary generator gets this episode's lava copied onto it by
        # GoalImageObservation, so the rendered goal is a state the live episode can
        # actually reach. Its own goal_pos/lava_pos here are placeholders, overwritten
        # per call; n_lava never applies because lava_pos is always assigned.
        env = wrappers.GoalImageObservation(
            env, lambda: make_lava_grid_env(size=config.env_size, goal_pos=(1, 1), lava_pos=[])
        )
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit // config.action_repeat)
    return wrappers.Dtype(env)

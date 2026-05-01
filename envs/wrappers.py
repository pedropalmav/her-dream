import gymnasium as gym
import numpy as np
import torch

import tools

def get_wrapper(env, wrapper_type):
    current = env
    while hasattr(current, "env"):
        if isinstance(current, wrapper_type):
            return current
        current = current.env
    return None


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
            # keep is_terminal as it is
            obs["is_last"] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self.action_space = space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)
    
    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class MultiOneHotAction(gym.Wrapper):
    def __init__(self, env, device):
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        super().__init__(env)
        self.index_low = torch.tensor(self.action_space.low, device=device)
        space = gym.spaces.Box(
            low=0, high=1, shape=self.env.action_space.nvec, dtype=np.float32
        )
        space.multi_discrete = True
        self.action_space = space

    def step(self, action1, action2, done):
        action1 = self.convert(action1)
        action2 = self.convert(action2)
        return self.env.step(action1, action2, done)

    def convert(self, action):
        now = 0
        indexes = []
        for dim in self.action_space.shape:
            index = torch.argmax(action[:, now : now + dim], dim=-1, keepdim=True)
            indexes.append(index)
            now += dim
        return torch.cat(indexes, dim=-1) + self.index_low


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class Dtype(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return tools.convert(obs), np.float32(rew), done, info

    def reset(self):
        return tools.convert(self.env.reset())


class GoalConditioned(gym.Wrapper):
    def __init__(self, env, config, text_encoder):
        super().__init__(env)

        self.stochastic_classes = config.stochastic_classes
        self.stochastic_rows = config.stochastic_rows
        self.goal_type = config.goal_type
        self.mission_text = config.mission_text
        self.text_encoder = text_encoder
        print("ENV encoder id:", id(self.text_encoder))
        

        if self.goal_type == "first_row":
            self.observation_space.spaces["goal"] = gym.spaces.MultiBinary(
                self.stochastic_classes
            )
            self.goal_index = config.get("goal_index", None)
        else:
            self.observation_space.spaces["goal"] = gym.spaces.MultiBinary(
                (self.stochastic_rows, self.stochastic_classes)
            )

    def reset(self):
        obs = self.env.reset()
        self._generate_goal()
        obs["goal"] = self.goal
        return obs

    # TODO: Evolve this method
    def _generate_goal(self):
        if self.goal_type == "first_row":
            print("using the fixed row")
            if self.goal_index:
                goal = np.zeros(self.stochastic_classes, dtype=np.float32)
                goal[self.goal_index] = 1.0
            else:
                goal = self._generate_row()
        elif self.mission_text:
            print("Generando goal desde encoder :o")
            goal = self._generate_goal_from_text()
        else:
            print("not using the text encoder ;c")
            goal = np.zeros(
                (self.stochastic_rows, self.stochastic_classes), dtype=np.float32
            )
            for i in range(self.stochastic_rows):
                goal[i] = self._generate_row()

        self.goal = goal

    def _generate_goal_from_text(self):
        """Genera un goal one-hot (S, K) a partir de una misión aleatoria."""
        import torch
        # 1. Obtenemos los tokens
        wrapper = get_wrapper(self.env, MissionGridWrapper)
        mission_tokens = wrapper.encoded_random_mission()

        # 2. Pasar por el text encoder: (1, 1, L, V) -> (1, 1, S, K)
        tokens = torch.from_numpy(mission_tokens).float().unsqueeze(0).unsqueeze(0)
        device = next(self.text_encoder.parameters()).device
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = self.text_encoder(tokens)  # (1, 1, S, K)

        logits = logits.squeeze(0).squeeze(0)   # (S, K)

        # 4. Samplear one-hot por fila
        probs = torch.softmax(logits, dim=-1)   # (S, K)
        indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (S,)

        goal = np.zeros(
            (self.stochastic_rows, self.stochastic_classes), dtype=np.float32
        )
        for i in range(self.stochastic_rows):
            goal[i, indices[i].item()] = 1.0

        return goal

    def _generate_row(self):
        row = np.zeros(self.stochastic_classes, dtype=np.float32)
        index = np.random.randint(0, self.stochastic_classes)
        row[index] = 1.0
        return row

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["goal"] = self.goal
        return obs, reward, done, info


class NoMission(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space.spaces.pop("mission", None)

    def observation(self, obs):
        obs.pop("mission", None)
        return obs


# TODO: Update this wrapper to replicate what DictConcat does
class OneHotDirection(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.spaces["direction"] = gym.spaces.Box(
            0, 1, shape=(4,), dtype=np.float32
        )

    def observation(self, obs):
        direction = obs.pop("direction")
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[direction] = 1.0
        obs["direction"] = one_hot
        return obs


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        env = NoMission(env)
        env = OneHotDirection(env)
        super().__init__(env)

        self.env.observation_space = gym.spaces.Dict(
            {
                **self.env.observation_space.spaces,
                "is_first": gym.spaces.Box(0, 1, (), bool),
                "is_last": gym.spaces.Box(0, 1, (), bool),
                "is_terminal": gym.spaces.Box(0, 1, (), bool),
            }
        )

    def reset(self):
        obs, _ = self.env.reset()
        return self._parse_observation(obs, is_first=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._parse_observation(
            obs,
            is_first=False,
            is_last=terminated or truncated,
            is_terminal=terminated,
        )
        return obs, reward, terminated or truncated, info

    def _parse_observation(self, obs, is_first=False, is_last=False, is_terminal=False):
        obs["is_first"] = is_first
        obs["is_last"] = is_last
        obs["is_terminal"] = is_terminal
        return obs


MAX_LEN = 500  # largo máximo del string
VOCAB = {c: i+1 for i, c in enumerate(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-:()")}
VOCAB["<pad>"] = 0
VOCAB_SIZE = len(VOCAB)  # 46 caracteres + padding

def encode_mission(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    text = text.lower()[:max_len]
    ids = [VOCAB[c] for c in text]
    ids += [0] * (max_len - len(ids))
    one_hot = np.zeros((max_len, VOCAB_SIZE), dtype=np.float32)
    for i, idx in enumerate(ids):
        one_hot[i, idx] = 1.0
    return one_hot


class MissionGridWrapper(gym.Wrapper):
    def __init__(self, env):
        env = OneHotDirection(env)
        super().__init__(env)

        self.env.observation_space = gym.spaces.Dict(
            {
                **self.env.observation_space.spaces,
                "mission": gym.spaces.Box(0, 1, (MAX_LEN, VOCAB_SIZE), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), bool),
                "is_last": gym.spaces.Box(0, 1, (), bool),
                "is_terminal": gym.spaces.Box(0, 1, (), bool),
            }
        )

    def reset(self):
        obs, _ = self.env.reset()
        return self._parse_observation(obs, is_first=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._parse_observation(
            obs,
            is_first=False,
            is_last=terminated or truncated,
            is_terminal=terminated,
        )
        return obs, reward, terminated or truncated, info

    def encoded_random_mission(self):
        return encode_mission(self.env.unwrapped.random_mission())

    def _parse_observation(self, obs, is_first=False, is_last=False, is_terminal=False):
        if "mission" in obs:
            obs["mission"] = encode_mission(obs["mission"])
        obs["is_first"] = is_first
        obs["is_last"] = is_last
        obs["is_terminal"] = is_terminal
        return obs
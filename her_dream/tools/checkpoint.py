import torch

# Actor-critic tensors used to live directly on `Dreamer`; they now live on its
# `ac` sub-module (`her_dream.actor_critic.ActorCritic`), which renamed every one
# of their state_dict keys. Checkpoints written before that refactor are still
# perfectly loadable — they just need their keys moved.
_LEGACY_AC_PREFIXES = {
    "actor.": "ac.actor.",
    "value.": "ac.value.",
    "_slow_value.": "ac._slow_value.",
    "_frozen_actor.": "ac._frozen_actor.",
    "_frozen_value.": "ac._frozen_value.",
    "_frozen_slow_value.": "ac._frozen_slow_value.",
    "return_ema.": "ac.return_ema.",
}

# `threshold_onehot` is a registered buffer with no dot in its key, so it needs
# an exact match rather than a prefix.
_LEGACY_AC_KEYS = {"threshold_onehot": "ac.threshold_onehot"}


def migrate_agent_state_dict(state_dict):
    """Rewrite a pre-`ActorCritic` agent state_dict to the current key layout.

    Idempotent: a migrated key (`ac.actor.…`) no longer starts with any legacy
    prefix, so re-running this is a no-op. Non-actor-critic tensors (encoder,
    rssm, reward/cont heads, decoder, text encoder) pass through untouched.
    """
    migrated = {}
    for key, value in state_dict.items():
        if key in _LEGACY_AC_KEYS:
            migrated[_LEGACY_AC_KEYS[key]] = value
            continue
        for old, new in _LEGACY_AC_PREFIXES.items():
            if key.startswith(old):
                migrated[new + key[len(old) :]] = value
                break
        else:
            migrated[key] = value
    return migrated


def recursively_collect_optim_state_dict(obj, path="", optimizers_state_dicts=None, visited=None):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update({k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr})
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(attr, new_path, optimizers_state_dicts, visited)
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)

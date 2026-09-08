import torch
import torch.nn as nn

from her_dream.tools.checkpoint import (
    migrate_agent_state_dict,
    recursively_collect_optim_state_dict,
    recursively_load_optim_state_dict,
)


class _Container:
    pass


class _ModelWithOptim(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.optim = torch.optim.SGD(self.linear.parameters(), lr=0.01)


class TestRecursivelyCollectOptimStateDict:
    def test_returns_empty_dict_for_object_with_no_optim(self):
        c = _Container()
        assert recursively_collect_optim_state_dict(c) == {}

    def test_optimizers_state_dicts_none_initializes_to_empty_dict(self):
        c = _Container()
        result = recursively_collect_optim_state_dict(c, optimizers_state_dicts=None)
        assert isinstance(result, dict)

    def test_visited_none_initializes_to_set(self):
        c = _Container()
        result = recursively_collect_optim_state_dict(c, visited=None)
        assert isinstance(result, dict)

    def test_cycle_detection_stops_infinite_loop(self):
        c = _Container()
        c.self_ref = c
        # Should terminate without RecursionError
        result = recursively_collect_optim_state_dict(c)
        assert result == {}

    def test_collects_optimizer_attribute(self):
        c = _Container()
        c.opt = torch.optim.Adam([torch.zeros(3)], lr=1e-3)
        result = recursively_collect_optim_state_dict(c)
        assert "opt" in result
        assert "state" in result["opt"]

    def test_top_level_path_is_just_name(self):
        c = _Container()
        c.opt = torch.optim.SGD([torch.zeros(2)], lr=0.01)
        result = recursively_collect_optim_state_dict(c)
        assert "opt" in result  # not ".opt"
        assert ".opt" not in result

    def test_nested_path_is_dotted(self):
        outer = _Container()
        inner = _Container()
        inner.opt = torch.optim.SGD([torch.zeros(2)], lr=0.01)
        outer.sub = inner
        result = recursively_collect_optim_state_dict(outer)
        assert "sub.opt" in result

    def test_nn_module_collects_optimizer_attr(self):
        m = _ModelWithOptim()
        result = recursively_collect_optim_state_dict(m)
        assert "optim" in result
        assert isinstance(result["optim"], dict)

    def test_explicit_visited_set_is_populated(self):
        c = _Container()
        visited = set()
        recursively_collect_optim_state_dict(c, visited=visited)
        assert id(c) in visited

    def test_passes_existing_dict_through(self):
        c = _Container()
        c.opt = torch.optim.SGD([torch.zeros(2)], lr=0.01)
        existing = {"prior_key": {"state": {}, "param_groups": []}}
        result = recursively_collect_optim_state_dict(c, optimizers_state_dicts=existing)
        assert "prior_key" in result
        assert "opt" in result

    def test_primitive_attribute_without_dict_is_skipped(self):
        c = _Container()
        c.num = 42  # int has no __dict__
        result = recursively_collect_optim_state_dict(c)
        assert result == {}


class TestRecursivelyLoadOptimStateDict:
    def test_loads_single_optimizer(self):
        class _ModelWithAdam(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.optim = torch.optim.Adam(self.linear.parameters(), lr=1e-3)

        model = _ModelWithAdam()
        # Take a step to populate per-parameter Adam state
        model.optim.zero_grad()
        model.linear(torch.ones(2)).sum().backward()
        model.optim.step()
        collected = recursively_collect_optim_state_dict(model)
        # Verify round-trip: after reload the state_dict matches
        recursively_load_optim_state_dict(model, collected)
        assert model.optim.state_dict() == collected["optim"]

    def test_loads_nested_optimizer(self):
        outer = _Container()
        inner = _Container()
        inner.opt = torch.optim.SGD([torch.zeros(2)], lr=0.01)
        outer.sub = inner
        collected = recursively_collect_optim_state_dict(outer)
        recursively_load_optim_state_dict(outer, collected)
        assert outer.sub.opt.state_dict() == collected["sub.opt"]


class TestMigrateAgentStateDict:
    """Pre-`ActorCritic` checkpoints used flat actor/critic keys.

    `Dreamer` now nests them under its `ac` sub-module, so every key the module
    owns moved. The migration is what keeps archived runs loadable.
    """

    def test_moves_every_actor_critic_prefix(self):
        legacy = {
            "actor.mlp.w": torch.ones(1),
            "value.mlp.w": torch.ones(1),
            "_slow_value.mlp.w": torch.ones(1),
            "_frozen_actor.mlp.w": torch.ones(1),
            "_frozen_value.mlp.w": torch.ones(1),
            "_frozen_slow_value.mlp.w": torch.ones(1),
            "return_ema.ema_vals": torch.ones(2),
        }
        assert set(migrate_agent_state_dict(legacy)) == {
            "ac.actor.mlp.w",
            "ac.value.mlp.w",
            "ac._slow_value.mlp.w",
            "ac._frozen_actor.mlp.w",
            "ac._frozen_value.mlp.w",
            "ac._frozen_slow_value.mlp.w",
            "ac.return_ema.ema_vals",
        }

    def test_moves_the_dotless_threshold_buffer(self):
        # `threshold_onehot` is a bare registered buffer, so no prefix matches it.
        assert set(migrate_agent_state_dict({"threshold_onehot": torch.ones(3)})) == {"ac.threshold_onehot"}

    def test_leaves_world_model_and_head_keys_alone(self):
        untouched = {
            "rssm.deter.w": torch.ones(1),
            "encoder.cnn.w": torch.ones(1),
            "_frozen_rssm.deter.w": torch.ones(1),
            "reward.last.weight": torch.ones(1),
            "cont.last.bias": torch.ones(1),
            "decoder.mlp.w": torch.ones(1),
            "text_encoder.gru.weight": torch.ones(1),
        }
        assert set(migrate_agent_state_dict(untouched)) == set(untouched)

    def test_slow_value_is_not_captured_by_the_value_prefix(self):
        # "_slow_value." must not be rewritten twice or land under "ac.value.".
        out = migrate_agent_state_dict({"_slow_value.mlp.w": torch.ones(1)})
        assert set(out) == {"ac._slow_value.mlp.w"}

    def test_preserves_the_tensors(self):
        tensor = torch.arange(3.0)
        assert migrate_agent_state_dict({"actor.mlp.w": tensor})["ac.actor.mlp.w"] is tensor

    def test_is_idempotent(self):
        legacy = {"actor.mlp.w": torch.ones(1), "rssm.w": torch.ones(1)}
        once = migrate_agent_state_dict(legacy)
        assert migrate_agent_state_dict(once) == once

    def test_a_current_state_dict_passes_through_unchanged(self):
        current = {"ac.actor.mlp.w": torch.ones(1), "ac.threshold_onehot": torch.ones(3)}
        assert set(migrate_agent_state_dict(current)) == set(current)

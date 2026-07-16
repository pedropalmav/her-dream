import contextlib
import io

import torch.nn as nn

from her_dream.tools.model_inspection import build_module_tree, print_module_tree, print_param_stats


def _capture(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args, **kwargs)
    return out.getvalue()


class TestBuildModuleTree:
    def test_linear_layer_param_count(self):
        m = nn.Linear(4, 8)
        tree = build_module_tree(m, "layer")
        # weight: 4*8=32, bias: 8 → total=40
        assert tree["total"] == 40

    def test_name_stored_correctly(self):
        m = nn.Linear(2, 2)
        tree = build_module_tree(m, "lin")
        assert tree["name"] == "lin"

    def test_no_params_module(self):
        m = nn.ReLU()
        tree = build_module_tree(m, "relu")
        assert tree["total"] == 0
        assert tree["params"] == {}
        assert tree["children"] == {}

    def test_nested_module_total_includes_children(self):
        m = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))
        tree = build_module_tree(m, "seq")
        child_total = sum(c["total"] for c in tree["children"].values())
        assert tree["total"] == child_total

    def test_children_keys_match_submodule_names(self):
        m = nn.Sequential(nn.Linear(2, 2))
        tree = build_module_tree(m)
        assert "0" in tree["children"]

    def test_direct_params_listed_in_params(self):
        m = nn.Linear(2, 4)
        tree = build_module_tree(m, "l")
        assert "weight" in tree["params"]
        assert "bias" in tree["params"]

    def test_default_module_name_is_empty_string(self):
        m = nn.Linear(2, 2)
        tree = build_module_tree(m)
        assert tree["name"] == ""


class TestPrintModuleTree:
    def test_branch1_empty_parent_path_uses_name(self):
        # full_path = name when parent_path is ""
        out = _capture(
            print_module_tree,
            {"name": "root", "params": {}, "children": {}, "total": 10},
        )
        assert "root" in out

    def test_branch2_both_parent_and_name_truthy(self):
        # full_path = f"{parent_path}/{name}"
        out = _capture(
            print_module_tree,
            {"name": "child", "params": {}, "children": {}, "total": 5},
            parent_path="parent",
        )
        assert "parent/child" in out

    def test_branch3_parent_truthy_name_empty(self):
        # full_path = parent_path when name is ""
        out = _capture(
            print_module_tree,
            {"name": "", "params": {}, "children": {}, "total": 5},
            parent_path="root",
        )
        assert "root" in out
        # Should NOT produce "root/" with a trailing slash
        assert "root/" not in out

    def test_prints_total_count(self):
        out = _capture(
            print_module_tree,
            {"name": "net", "params": {}, "children": {}, "total": 1_234_567},
        )
        assert "1,234,567" in out

    def test_sorts_children_descending_by_total(self):
        info = {
            "name": "top",
            "params": {},
            "children": {
                "small": {"name": "small", "params": {}, "children": {}, "total": 1},
                "large": {"name": "large", "params": {}, "children": {}, "total": 100},
            },
            "total": 101,
        }
        out = _capture(print_module_tree, info)
        assert out.index("large") < out.index("small")

    def test_param_nodes_appear_in_output(self):
        info = {
            "name": "layer",
            "params": {"weight": 32, "bias": 8},
            "children": {},
            "total": 40,
        }
        out = _capture(print_module_tree, info)
        assert "weight" in out
        assert "bias" in out

    def test_indent_increases_for_children(self):
        m = nn.Linear(2, 4)
        tree = build_module_tree(m, "layer")
        out = _capture(print_module_tree, tree)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        # Parent line has less leading whitespace than child lines
        parent_indent = len(lines[0]) - len(lines[0].lstrip())
        child_indents = [len(ln) - len(ln.lstrip()) for ln in lines[1:]]
        assert all(ci > parent_indent for ci in child_indents)


class TestPrintParamStats:
    def test_prints_header(self, capsys):
        m = nn.Linear(2, 2)
        print_param_stats(m)
        out = capsys.readouterr().out
        assert "Parameter" in out
        assert "Mean" in out

    def test_trainable_params_appear_in_output(self, capsys):
        m = nn.Linear(4, 4)
        print_param_stats(m)
        out = capsys.readouterr().out
        assert "weight" in out

    def test_non_trainable_params_skipped(self, capsys):
        m = nn.Linear(4, 4)
        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)
        print_param_stats(m)
        out = capsys.readouterr().out
        assert "weight" not in out

    def test_dot_replaced_by_slash_in_name(self, capsys):
        m = nn.Sequential(nn.Linear(2, 2))
        print_param_stats(m)
        out = capsys.readouterr().out
        assert "0/weight" in out
        assert "0.weight" not in out

    def test_model_with_no_trainable_params_prints_header_only(self, capsys):
        m = nn.Linear(2, 2)
        for p in m.parameters():
            p.requires_grad_(False)
        print_param_stats(m)
        out = capsys.readouterr().out
        assert "Parameter" in out

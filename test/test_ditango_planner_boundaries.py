from types import SimpleNamespace

import torch


class DummyGroup:
    def __init__(self, group_size=1, rank_in_group=0):
        self.group_size = group_size
        self.rank_in_group = rank_in_group

    def all_reduce(self, tensor, op=None):
        return tensor


def _patch_planner_groups(monkeypatch, group_size=4):
    import chitu_diffusion.ditango.planner as planner_mod

    cp_group = DummyGroup(group_size=group_size, rank_in_group=0)
    cfg_group = DummyGroup(group_size=1, rank_in_group=0)
    world_group = DummyGroup(group_size=1, rank_in_group=0)
    monkeypatch.setattr(planner_mod, "get_cp_group", lambda: cp_group)
    monkeypatch.setattr(planner_mod, "get_cfg_group", lambda: cfg_group)
    monkeypatch.setattr(planner_mod, "get_world_group", lambda: world_group)
    monkeypatch.setattr(planner_mod, "should_log_info_on_rank", lambda: False)
    return planner_mod


def _make_planner(monkeypatch, cache_ratio=0.5, power=1.0 / 3.0):
    planner_mod = _patch_planner_groups(monkeypatch)
    task = SimpleNamespace(
        task_id="task",
        req=SimpleNamespace(params=SimpleNamespace(num_inference_steps=20)),
    )
    planner = planner_mod.DiTangoPlanner(
        task=task,
        cache_ratio=cache_ratio,
        warmup_steps=1,
        cooldown_steps=1,
        intra_group_size_limit=1,
        tau_max=8,
        curvature_interval_power=power,
    )
    planner.total_layers = 2
    return planner


def test_power_law_intervals_preserve_curvature_order(monkeypatch):
    planner = _make_planner(monkeypatch, cache_ratio=0.5)
    curv = torch.tensor([[100.0, 1.0, 0.1, 0.01]])

    intervals = planner._compute_interval_plan(curv)

    assert intervals[0, 0] <= intervals[0, 1]
    assert intervals[0, 1] <= intervals[0, 2]
    assert intervals[0, 2] <= intervals[0, 3]
    assert int(intervals.min()) >= 1
    assert int(intervals.max()) <= planner.tau_max


def test_power_parameter_increases_interval_contrast(monkeypatch):
    curv = torch.tensor([[16.0, 1.0]])
    low_power = _make_planner(monkeypatch, cache_ratio=0.75, power=0.1)
    high_power = _make_planner(monkeypatch, cache_ratio=0.75, power=1.0)

    low_intervals = low_power._compute_interval_plan(curv)
    high_intervals = high_power._compute_interval_plan(curv)

    low_gap = int(low_intervals[0, 1]) - int(low_intervals[0, 0])
    high_gap = int(high_intervals[0, 1]) - int(high_intervals[0, 0])
    assert high_gap >= low_gap


def test_anchor_full_output_drift_builds_group_curvature(monkeypatch):
    planner = _make_planner(monkeypatch)
    planner.last_anchor_step["pos"] = 1
    planner.full_output_compress["pos_0"] = torch.ones((1, 4, 2))
    planner.prev_anchor_full_output_compress["pos_0"] = torch.zeros((1, 4, 2))
    planner.full_output_compress["pos_1"] = torch.full((1, 4, 2), 2.0)
    planner.prev_anchor_full_output_compress["pos_1"] = torch.zeros((1, 4, 2))
    for layer in range(2):
        for group in range(4):
            planner.group_meta[f"pos_{layer}_group_{group}"] = {"weight": 0.25}

    curvature, weights = planner._build_anchor_matrices("pos", step=3)

    assert torch.allclose(weights, torch.full((2, 4), 0.25))
    assert torch.all(curvature[1] > curvature[0])


def test_selective_plan_uses_interval_age(monkeypatch):
    planner = _make_planner(monkeypatch)
    planner.group_intervals["pos"] = torch.tensor([[1, 2, 3, 4], [2, 2, 4, 4]])
    planner.last_anchor_step["pos"] = 5

    planner._build_plan_from_intervals(step=7, branch_key="pos")

    expected = torch.tensor(
        [[True, True, False, False], [True, True, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(planner.ditango_plan["pos"], expected)


def test_initial_anchor_steps_are_inside_warmup(monkeypatch):
    planner_mod = _patch_planner_groups(monkeypatch)
    task = SimpleNamespace(
        task_id="task",
        req=SimpleNamespace(params=SimpleNamespace(num_inference_steps=20)),
    )
    planner = planner_mod.DiTangoPlanner(
        task=task,
        cache_ratio=0.5,
        warmup_steps=7,
        cooldown_steps=3,
        intra_group_size_limit=1,
    )

    assert planner.anchor_step_list["pos"] == [5, 6]
    assert planner.anchor_step_list["neg"] == [5, 6]


def test_next_anchor_is_not_scheduled_inside_cooldown(monkeypatch):
    planner = _make_planner(monkeypatch)
    planner.group_intervals["pos"] = torch.full((2, 4), 8, dtype=torch.long)

    planner._schedule_next_anchor(step=3, target_branches=("pos",))

    assert planner.anchor_step_list["pos"] == [0, 1]

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


def _make_planner(monkeypatch, cache_ratio=0.5, power=1.0 / 3.0, locality_group_compute_boost=0.0):
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
        locality_group_compute_boost=locality_group_compute_boost,
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


def test_cache_ratio_targets_end_to_end_reuse_after_forced_steps(monkeypatch):
    planner = _make_planner(monkeypatch, cache_ratio=0.4)
    planner.warmup_steps = 5
    planner.cooldown_steps = 5
    planner.anchor_step_list["pos"].append(10)
    planner.anchor_step_list["neg"].append(10)

    assert planner._selective_cache_ratio_target() == 0.4 * 20 / 9


def test_default_group_limit_uses_per_cp_rank_groups(monkeypatch):
    planner_mod = _patch_planner_groups(monkeypatch, group_size=4)
    task = SimpleNamespace(
        task_id="task",
        req=SimpleNamespace(params=SimpleNamespace(num_inference_steps=20)),
    )
    planner = planner_mod.DiTangoPlanner(task=task)

    assert planner.intra_group_size_limit == 1
    assert planner.group_num == 4


def test_power_parameter_increases_interval_contrast(monkeypatch):
    curv = torch.tensor([[16.0, 1.0]])
    low_power = _make_planner(monkeypatch, cache_ratio=0.75, power=0.1)
    high_power = _make_planner(monkeypatch, cache_ratio=0.75, power=1.0)

    low_intervals = low_power._compute_interval_plan(curv)
    high_intervals = high_power._compute_interval_plan(curv)

    low_gap = int(low_intervals[0, 1]) - int(low_intervals[0, 0])
    high_gap = int(high_intervals[0, 1]) - int(high_intervals[0, 0])
    assert high_gap >= low_gap


def test_locality_boost_computes_front_groups_more_often(monkeypatch):
    planner = _make_planner(
        monkeypatch,
        cache_ratio=0.6,
        power=0.0,
        locality_group_compute_boost=1.0,
    )
    curv = torch.ones((1, 4))

    intervals = planner._compute_interval_plan(curv)

    assert intervals[0, 0] <= intervals[0, 2]
    assert intervals[0, 1] <= intervals[0, 2]
    assert intervals[0, 1] <= intervals[0, 3]


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

    planner._schedule_next_anchor(step=15, target_branches=("pos",))

    assert planner.anchor_step_list["pos"] == [0, 1]


def test_groupwise_stagger_overrides_only_layer_band(monkeypatch):
    planner = _make_planner(monkeypatch)
    planner.total_layers = 4
    planner.groupwise_stagger_period = 3
    planner.groupwise_stagger_fresh_count = 1
    planner.groupwise_stagger_layer_start = 1
    planner.groupwise_stagger_layer_end = 2
    planner.groupwise_keep_local = True
    planner.groupwise_force_tail_full_layers = 1
    plan = torch.zeros((4, 4), dtype=torch.bool)
    plan[0, 3] = True

    result = planner._apply_groupwise_stagger_plan(plan, step=3)

    assert torch.equal(result[0], torch.tensor([False, False, False, True]))
    assert torch.equal(result[3], torch.ones(4, dtype=torch.bool))
    assert result[1, 0]
    assert result[2, 0]
    assert result[1, 2]
    assert result[2, 2]
    assert int(result[1].sum().item()) == 2
    assert int(result[2].sum().item()) == 2


def test_anchor_interval_is_fixed_for_groupwise(monkeypatch):
    planner = _make_planner(monkeypatch)
    planner.anchor_interval = 4
    planner.group_intervals["pos"] = torch.full((2, 4), 8, dtype=torch.long)

    planner._schedule_next_anchor(step=3, target_branches=("pos",))

    assert 7 in planner.anchor_step_list["pos"]


def test_kv_cache_is_stored_and_discarded_with_group_state(monkeypatch):
    planner = _make_planner(monkeypatch)
    import chitu_diffusion.ditango.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_timestep", lambda: 2)
    k = torch.ones((1, 2, 3, 4))
    v = torch.full((1, 2, 3, 4), 2.0)
    from chitu_diffusion.ditango.state import AttentionState

    state = AttentionState(out=torch.zeros((1, 2, 3, 4)), lse=torch.zeros((1, 3, 2)))

    planner.store(layer_id=1, group_id=2, group_state=state)
    planner.store_kv(layer_id=1, group_id=2, k=k, v=v)

    cached = planner.reuse_kv(layer_id=1, group_id=2)
    assert cached is not None
    cached_k, cached_v, cached_cu = cached
    assert torch.equal(cached_k, k)
    assert torch.equal(cached_v, v)
    assert cached_cu is None

    planner.discard_group_state(layer_id=1, group_id=2)

    assert planner.reuse(layer_id=1, group_id=2).is_empty()
    assert planner.reuse_kv(layer_id=1, group_id=2) is None


def test_compressed_state_cache_has_separate_lifecycle(monkeypatch):
    planner = _make_planner(monkeypatch)
    from chitu_diffusion.ditango.state import AttentionState

    group_state = AttentionState(out=torch.ones((1, 2, 3, 4)), lse=torch.zeros((1, 2, 3, 1)))
    compressed_state = AttentionState(out=torch.full((1, 2, 3, 4), 2.0), lse=torch.ones((1, 2, 3, 1)))

    planner.state_cache[planner.get_store_key(layer_id=1, group_id=2)] = group_state
    planner.store_compressed(layer_id=1, tag="nonlocal", state=compressed_state)
    planner.discard_group_state(layer_id=1, group_id=2)

    assert planner.reuse(layer_id=1, group_id=2).is_empty()
    assert torch.equal(planner.reuse_compressed(layer_id=1, tag="nonlocal").out, compressed_state.out)

    planner.discard_compressed_state(layer_id=1, tag="nonlocal")
    assert planner.reuse_compressed(layer_id=1, tag="nonlocal").is_empty()

    planner.store_compressed(layer_id=1, tag="nonlocal", state=compressed_state)
    planner.reset_state()
    assert planner.reuse_compressed(layer_id=1, tag="nonlocal").is_empty()


def test_anchor_stores_local_and_compressed_nonlocal_for_local_only(monkeypatch):
    planner = _make_planner(monkeypatch)
    planner.intra_group_size_limit = 1
    planner.effective_group_size = 1
    planner.group_num = 4
    planner.groupwise_local_expand = 0
    planner.groupwise_reuse_stale_kv = False
    planner.groupwise_stagger_period = 0
    planner.groupwise_stagger_fresh_count = 0
    planner.groupwise_topk_mode = "none"
    planner.groupwise_extra_topk = 0

    import chitu_diffusion.ditango.runtime as runtime_mod
    import chitu_diffusion.ditango.planner as planner_mod

    monkeypatch.setattr(runtime_mod, "get_ditango_planner", lambda: planner)
    monkeypatch.setattr(planner_mod, "get_timestep", lambda: 2)
    monkeypatch.setattr(runtime_mod, "async_ring_p2p_commit", lambda group, tensors, src_rank, dst_rank: tensors)
    monkeypatch.setattr(runtime_mod, "async_ring_p2p_wait_and_update", lambda group, tensors: tensors)

    attn = object.__new__(runtime_mod.DitangoAttention)
    attn.group = DummyGroup(group_size=4, rank_in_group=0)
    attn.cp_size = 4
    attn.rank_in_cp = 0
    attn.layer_id = 0
    attn.intra_group_size = 1
    attn.ulysses_size = 1

    calls = {"idx": 0}

    def fake_backend(q, k, v, **kwargs):
        calls["idx"] += 1
        value = float(calls["idx"])
        out = torch.full((1, 2, 1, 1), value)
        lse = torch.full((1, 1, 2), value)
        return out, lse, None

    attn.attn_backend = fake_backend
    q = torch.zeros((1, 2, 1, 1))
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)

    final_state, group_states, local_state = attn._anchor_step_attn(q, k, v, is_varlen=False, is_anchor=True)

    assert not final_state.is_empty()
    assert not local_state.is_empty()
    assert set(group_states.keys()) == {0, 1, 2, 3}
    assert not planner.reuse(layer_id=0, group_id=0).is_empty()
    for group_id in (1, 2, 3):
        assert planner.reuse(layer_id=0, group_id=group_id).is_empty()
    assert not planner.reuse_compressed(layer_id=0, tag="nonlocal").is_empty()

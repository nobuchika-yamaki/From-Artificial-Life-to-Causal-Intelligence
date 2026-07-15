#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_DARCA_V24_INTELLIGENCE_EMERGENCE_LEVEL_BATTERY_READOUT_FIXED.py
=========================================================

Corrected causal capability battery for DARCA v2.4 signed action readout.

The scientific design is unchanged from the level battery:
  1. current_sufficient
  2. history_required
  3. hidden_reversal
  4. transfer
  5. action_independent_null

The implementation is reorganized for computational efficiency and stricter
pairing:
  - one worker import of the supplied DARCA source and integrated core;
  - one bundle per seed x regime, containing every task and condition;
  - one shared external schedule per task/phase inside each bundle;
  - continuous native model RNG streams (no per-step RNG reconstruction);
  - online metric accumulation without step dictionaries or action.to_row();
  - one integrated outer-layer template per bundle, cloned across conditions;
  - official structural lesions audited by declared flag differences and deterministic
    same-condition initialization, rather than an invalid cross-lesion state-hash equality;
  - vectorized, batched bootstrap and sign-flip inference;
  - resumable bundle files with schema and experiment-hash validation;
  - measured probe-bundle runtime and projected wall time.

No weighted intelligence score, result-dependent task adjustment, automatic
pass/fail label, or selective reporting is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import shutil
import sys
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_NAME = "15_DARCA_V24_INTELLIGENCE_EMERGENCE_LEVEL_BATTERY_READOUT_FIXED.py"
SCHEMA_VERSION = 5
TASKS = (
    "current_sufficient",
    "history_required",
    "hidden_reversal",
    "transfer",
    "action_independent_null",
)
CONDITION_NAMES = (
    "Full",
    "Random_Readout",
    "No_MetaAutonomy",
    "No_CausalInference",
    "No_Agency",
    "No_Memory",
    "Symmetric_Delay",
    "No_TemporalMembrane",
    "No_DissipativeCore",
    "No_ViabilityAutonomy",
)
ACTION_NAMES_FALLBACK = {
    0: "REGULATE",
    1: "PROBE_PLUS",
    2: "PROBE_MINUS",
    3: "INHIBIT",
    4: "EXPRESS",
}
PRIMARY_METRICS = (
    "probe_opportunity_count",
    "probe_count",
    "correct_probe_count",
    "incorrect_probe_count",
    "correct_probe_rate",
    "incorrect_probe_rate",
    "probe_balance",
    "cumulative_regret",
    "direction_readout_accuracy",
    "direction_readout_coverage",
    "direction_readout_abs_mean",
    "causal_direction_accuracy",
    "causal_direction_coverage",
    "causal_direction_abs_mean",
    "probe_readout_agreement_rate",
    "sensory_sign_accuracy",
    "sensory_sign_coverage",
    "outcome_sum",
    "damage_sum",
    "benefit_sum",
    "h_auc",
    "h_final",
    "autonomy_mean",
    "identity_mean",
    "causal_confidence_mean",
    "agency_abs_mean",
    "memory_force_mean",
    "prediction_error_mean",
    "terminal_fraction",
    "motor_alignment_rate",
)

# Worker-local resources. They are initialized once per process.
_WORKER_DARCA: Any = None
_WORKER_CORE: Any = None
_WORKER_CONDITIONS: Dict[str, Any] = {}
_WORKER_DARCA_PATH = ""
_WORKER_CORE_PATH = ""
_WORKER_IMPORT_COUNT = 0


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def stable_int(*parts: Any) -> int:
    raw = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    raw = json.dumps(
        obj,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple, set)):
                    value = json.dumps(value, sort_keys=True, default=str)
                out[key] = value
            writer.writerow(out)


def bh_fdr(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 1.0
    for reverse_rank, idx in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        running = min(running, float(p[idx]) * m / rank)
        q[idx] = min(1.0, running)
    return q.tolist()


@dataclass(frozen=True)
class Plan:
    seeds: int
    steps: int
    history_steps: int
    eval_steps: int
    reversal_step: int
    workers_cap: int
    bootstraps: int
    permutations: int
    integrated: bool


def plan_defaults(name: str) -> Plan:
    if name == "smoke":
        return Plan(1, 40, 16, 16, 20, 4, 200, 300, True)
    if name == "quick":
        return Plan(8, 500, 200, 200, 250, 8, 1000, 2000, True)
    if name == "main":
        return Plan(32, 1000, 400, 400, 500, 12, 5000, 10000, True)
    raise ValueError(f"Unknown plan: {name}")


@dataclass(frozen=True)
class Regime:
    delay: int
    noise: float
    coupling: float

    @property
    def id(self) -> str:
        return f"d{self.delay}_n{self.noise:.3f}_c{self.coupling:.2f}"


@dataclass(frozen=True)
class Bundle:
    darca_file: str
    agent_core: str
    seed_id: int
    base_seed: int
    regime: Regime
    steps: int
    history_steps: int
    eval_steps: int
    reversal_step: int
    run_native: bool
    run_integrated: bool
    experiment_hash: str


def build_regimes(plan_name: str) -> List[Regime]:
    if plan_name == "smoke":
        return [Regime(4, 0.035, 0.45)]
    if plan_name == "quick":
        return [Regime(4, 0.035, 0.45), Regime(12, 0.060, 0.80)]
    return [Regime(d, n, c) for d in (4, 8, 12) for n in (0.035, 0.060) for c in (0.45, 0.80)]


def extract_conditions(mod: Any) -> Dict[str, Any]:
    if hasattr(mod, "build_conditions"):
        values = list(mod.build_conditions("all"))
        out = {str(getattr(c, "name", "")): c for c in values}
    else:
        Condition = getattr(mod, "Condition")
        templates = {
            "Full": {},
            "Random_Readout": {"direction_readout_enabled": False},
            "No_MetaAutonomy": {"meta_enabled": False},
            "No_CausalInference": {"causal_enabled": False},
            "No_Agency": {"agency_enabled": False},
            "No_Memory": {"memory_enabled": False},
            "Symmetric_Delay": {"delta_tau_override": 0},
            "No_TemporalMembrane": {"temporal_enabled": False},
            "No_DissipativeCore": {"dissipative_enabled": False},
            "No_ViabilityAutonomy": {"viability_enabled": False},
        }
        allowed = set(inspect.signature(Condition).parameters)
        out = {}
        for name, kwargs in templates.items():
            clean = {k: v for k, v in kwargs.items() if k in allowed}
            out[name] = Condition(name, **clean)
    missing = [name for name in CONDITION_NAMES if name not in out]
    if missing:
        raise RuntimeError(f"DARCA source does not provide required conditions: {missing}")
    return out


EXPECTED_CONDITION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "Full": {},
    "Random_Readout": {"direction_readout_enabled": False},
    "No_MetaAutonomy": {"meta_enabled": False},
    "No_CausalInference": {"causal_enabled": False},
    "No_Agency": {"agency_enabled": False},
    "No_Memory": {"memory_enabled": False},
    "Symmetric_Delay": {"delta_tau_override": 0},
    "No_TemporalMembrane": {"temporal_enabled": False},
    "No_DissipativeCore": {"dissipative_enabled": False},
    "No_ViabilityAutonomy": {"viability_enabled": False},
}


def condition_config(condition: Any) -> Dict[str, Any]:
    """Return the public condition configuration without its display name.

    Official DARCA lesions are structural interventions. They legitimately create
    different module shapes, enabled flags, recurrent gains, and therefore different
    complete initial-state hashes. Pairing is established by the same seed and the
    same external schedules, not by forcing structurally different agents to have an
    identical serialized state.
    """
    if is_dataclass(condition):
        raw = asdict(condition)
    elif hasattr(condition, "__dict__"):
        raw = {k: v for k, v in vars(condition).items() if not k.startswith("_")}
    else:
        raw = {
            key: getattr(condition, key)
            for key in (
                "name", "meta_enabled", "causal_enabled", "agency_enabled",
                "memory_enabled", "temporal_enabled", "dissipative_enabled",
                "viability_enabled", "delta_tau_override",
            )
            if hasattr(condition, key)
        }
    raw.pop("name", None)
    return raw


def audit_official_condition_definitions(conditions: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify that every official lesion differs from Full only as declared."""
    full = condition_config(conditions["Full"])
    signatures: Dict[str, Dict[str, Any]] = {}
    observed_differences: Dict[str, Dict[str, Any]] = {}
    for name in CONDITION_NAMES:
        cfg = condition_config(conditions[name])
        signatures[name] = cfg
        keys = sorted(set(full) | set(cfg))
        diffs = {k: cfg.get(k) for k in keys if cfg.get(k) != full.get(k)}
        expected = EXPECTED_CONDITION_OVERRIDES[name]
        if diffs != expected:
            raise RuntimeError(
                f"Official condition {name} has unexpected configuration differences: "
                f"observed={diffs}, expected={expected}"
            )
        observed_differences[name] = diffs
    return {
        "full_signature": full,
        "condition_signatures": signatures,
        "observed_differences_from_full": observed_differences,
    }


def make_params(mod: Any, regime: Regime) -> Any:
    Params = getattr(mod, "Params")
    params = Params()
    updates: Dict[str, Any] = {}
    if hasattr(params, "causal_max_delay"):
        updates["causal_max_delay"] = max(12, int(regime.delay))
    if is_dataclass(params) and updates:
        params = replace(params, **updates)
    else:
        for key, value in updates.items():
            try:
                setattr(params, key, value)
            except Exception:
                pass
    return params


def patch_core_loader(core: Any, darca_mod: Any) -> bool:
    """Reuse the already imported exact DARCA module inside the integrated core.

    This changes only module loading. It does not replace the Agent class or any
    model computation.
    """
    wrapper = getattr(core, "DarcaV24Wrapper", None)
    if wrapper is None or not hasattr(wrapper, "_load_module"):
        return False

    def cached_loader(_path: str) -> Any:
        return darca_mod

    wrapper._load_module = staticmethod(cached_loader)
    return True


def worker_init(darca_file: str, agent_core: str) -> None:
    global _WORKER_DARCA, _WORKER_CORE, _WORKER_CONDITIONS
    global _WORKER_DARCA_PATH, _WORKER_CORE_PATH, _WORKER_IMPORT_COUNT
    darca_path = str(Path(darca_file).expanduser().resolve())
    core_path = str(Path(agent_core).expanduser().resolve())
    if (
        _WORKER_DARCA is not None
        and _WORKER_CORE is not None
        and _WORKER_DARCA_PATH == darca_path
        and _WORKER_CORE_PATH == core_path
    ):
        return
    pid = os.getpid()
    _WORKER_DARCA = import_module(f"darca_fast_worker_{pid}", Path(darca_path))
    _WORKER_CORE = import_module(f"integrated_fast_worker_{pid}", Path(core_path))
    _WORKER_CONDITIONS = extract_conditions(_WORKER_DARCA)
    patch_core_loader(_WORKER_CORE, _WORKER_DARCA)
    _WORKER_DARCA_PATH = darca_path
    _WORKER_CORE_PATH = core_path
    _WORKER_IMPORT_COUNT += 1


def clone_without_module(obj: Any) -> Any:
    holders: List[Tuple[Any, str, Any]] = []
    stack = [obj]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if hasattr(current, "module"):
            try:
                holders.append((current, "module", getattr(current, "module")))
                setattr(current, "module", None)
            except Exception:
                pass
        if hasattr(current, "__dict__"):
            for value in vars(current).values():
                if hasattr(value, "__dict__"):
                    stack.append(value)
    try:
        cloned = copy.deepcopy(obj)
    finally:
        for holder, name, value in holders:
            setattr(holder, name, value)

    def restore(original: Any, copied: Any, visited: set[Tuple[int, int]]) -> None:
        key = (id(original), id(copied))
        if key in visited:
            return
        visited.add(key)
        if hasattr(original, "module") and hasattr(copied, "module") and getattr(copied, "module", None) is None:
            try:
                setattr(copied, "module", getattr(original, "module"))
            except Exception:
                pass
        if hasattr(original, "__dict__") and hasattr(copied, "__dict__"):
            for name, value in vars(original).items():
                copied_value = vars(copied).get(name)
                if hasattr(value, "__dict__") and hasattr(copied_value, "__dict__"):
                    restore(value, copied_value, visited)

    restore(obj, cloned, set())
    return cloned


def jsonable_state(obj: Any, depth: int = 0, seen: Optional[set[int]] = None) -> Any:
    if seen is None:
        seen = set()
    if depth > 8:
        return "<max-depth>"
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return jsonable_state(obj.item(), depth + 1, seen)
    if isinstance(obj, np.ndarray):
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sha": hashlib.sha256(obj.tobytes()).hexdigest(),
        }
    if isinstance(obj, np.random.Generator):
        return {"rng": obj.bit_generator.state}
    if isinstance(obj, Mapping):
        return {
            str(k): jsonable_state(v, depth + 1, seen)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [jsonable_state(v, depth + 1, seen) for v in obj]
    if id(obj) in seen:
        return "<cycle>"
    seen.add(id(obj))
    if is_dataclass(obj):
        return {
            f.name: jsonable_state(getattr(obj, f.name), depth + 1, seen)
            for f in fields(obj)
            if f.name != "name"
        }
    if hasattr(obj, "__dict__"):
        excluded = {"condition", "module", "darca_file"}
        return {
            k: jsonable_state(v, depth + 1, seen)
            for k, v in sorted(vars(obj).items())
            if k not in excluded and not k.startswith("__")
        }
    return repr(obj)


def state_hash(obj: Any) -> str:
    return canonical_hash(jsonable_state(obj))


def action_name(row: Mapping[str, Any], mod: Any) -> str:
    name = row.get("action_name")
    if name is not None:
        return str(name)
    aid = int(safe_float(row.get("action_id"), -1))
    names = getattr(mod, "ACTION_NAMES", ACTION_NAMES_FALLBACK)
    return str(names.get(aid, f"ACTION_{aid}"))


def action_sign(name: str) -> int:
    if "PROBE_PLUS" in name:
        return 1
    if "PROBE_MINUS" in name:
        return -1
    return 0


def first_value(mapping: Mapping[str, Any], names: Sequence[str], default: float = float("nan")) -> float:
    for name in names:
        if name in mapping:
            value = safe_float(mapping.get(name), default)
            if math.isfinite(value):
                return value
    return default


def action_value(action: Any, names: Sequence[str], default: float = float("nan")) -> float:
    for name in names:
        if hasattr(action, name):
            value = safe_float(getattr(action, name), default)
            if math.isfinite(value):
                return value
    debug = getattr(action, "debug", None)
    if isinstance(debug, Mapping):
        return first_value(debug, names, default)
    return default


def env_info_template(regime: Regime) -> Dict[str, float]:
    return {
        "external_shock": 0.0,
        "d_dyn": float(regime.delay),
        "coupling_t": float(regime.coupling),
        "sigma_t": float(regime.noise),
    }


def delayed_outcome_for_action(rule: int, sign: int, magnitude: float) -> float:
    if sign == 0:
        return 0.0
    return magnitude if sign == rule else -magnitude


def schedule_noise(base_seed: int, seed_id: int, task: str, regime: Regime, phase: str, n: int) -> np.ndarray:
    rng = np.random.default_rng(
        stable_int(base_seed, seed_id, task, regime.id, phase, "external_noise") & 0xFFFFFFFF
    )
    return rng.normal(0.0, regime.noise, size=n).astype(np.float64, copy=False)


def schedule_null_outcomes(base_seed: int, seed_id: int, regime: Regime, n: int) -> np.ndarray:
    signs = np.ones(n, dtype=np.int8)
    signs[: n // 2] = -1
    rng = np.random.default_rng(stable_int(base_seed, seed_id, regime.id, "null") & 0xFFFFFFFF)
    rng.shuffle(signs)
    return signs


class MetricAccumulator:
    __slots__ = (
        "n", "probes", "correct", "incorrect", "plus", "minus",
        "outcome_sum", "damage_sum", "benefit_sum", "h_sum", "h_count",
        "h_final", "autonomy_sum", "autonomy_n", "identity_sum", "identity_n",
        "causal_sum", "causal_n", "agency_sum", "agency_n", "memory_sum",
        "memory_n", "pred_sum", "pred_n", "terminal_sum", "terminal_n",
        "motor_sum", "motor_n", "direction_available", "direction_correct",
        "direction_abs_sum", "causal_dir_available", "causal_dir_correct",
        "causal_dir_abs_sum", "direction_probe_n", "direction_probe_agree",
        "sensory_available", "sensory_correct",
    )

    def __init__(self) -> None:
        self.n = 0
        self.probes = 0
        self.correct = 0
        self.incorrect = 0
        self.plus = 0
        self.minus = 0
        self.outcome_sum = 0.0
        self.damage_sum = 0.0
        self.benefit_sum = 0.0
        self.h_sum = 0.0
        self.h_count = 0
        self.h_final = float("nan")
        self.autonomy_sum = 0.0
        self.autonomy_n = 0
        self.identity_sum = 0.0
        self.identity_n = 0
        self.causal_sum = 0.0
        self.causal_n = 0
        self.agency_sum = 0.0
        self.agency_n = 0
        self.memory_sum = 0.0
        self.memory_n = 0
        self.pred_sum = 0.0
        self.pred_n = 0
        self.terminal_sum = 0.0
        self.terminal_n = 0
        self.motor_sum = 0.0
        self.motor_n = 0
        self.direction_available = 0
        self.direction_correct = 0
        self.direction_abs_sum = 0.0
        self.causal_dir_available = 0
        self.causal_dir_correct = 0
        self.causal_dir_abs_sum = 0.0
        self.direction_probe_n = 0
        self.direction_probe_agree = 0
        self.sensory_available = 0
        self.sensory_correct = 0

    @staticmethod
    def _add(value: float, total_name: str, count_name: str, obj: "MetricAccumulator") -> None:
        if math.isfinite(value):
            setattr(obj, total_name, getattr(obj, total_name) + value)
            setattr(obj, count_name, getattr(obj, count_name) + 1)

    def update(
        self,
        rule: int,
        probe_sign: int,
        delivered: float,
        h: float,
        autonomy: float,
        identity: float,
        causal: float,
        agency: float,
        memory: float,
        prediction_error: float,
        terminal: float,
        motor_alignment: float = float("nan"),
        direction_score: float = float("nan"),
        causal_direction_score: float = float("nan"),
        sensory_score: float = float("nan"),
    ) -> None:
        self.n += 1
        if probe_sign != 0:
            self.probes += 1
            if probe_sign > 0:
                self.plus += 1
            else:
                self.minus += 1
            if probe_sign == rule:
                self.correct += 1
            else:
                self.incorrect += 1
        if math.isfinite(direction_score) and abs(direction_score) > 1e-15:
            self.direction_available += 1
            self.direction_abs_sum += abs(direction_score)
            direction_sign = 1 if direction_score > 0.0 else -1
            if direction_sign == rule:
                self.direction_correct += 1
            if probe_sign != 0:
                self.direction_probe_n += 1
                if direction_sign == probe_sign:
                    self.direction_probe_agree += 1
        if math.isfinite(causal_direction_score) and abs(causal_direction_score) > 1e-15:
            self.causal_dir_available += 1
            self.causal_dir_abs_sum += abs(causal_direction_score)
            causal_sign = 1 if causal_direction_score > 0.0 else -1
            if causal_sign == rule:
                self.causal_dir_correct += 1
        if math.isfinite(sensory_score) and abs(sensory_score) > 1e-15:
            self.sensory_available += 1
            sensory_sign = 1 if sensory_score > 0.0 else -1
            if sensory_sign == rule:
                self.sensory_correct += 1
        self.outcome_sum += delivered
        self.damage_sum += max(0.0, -delivered)
        self.benefit_sum += max(0.0, delivered)
        if math.isfinite(h):
            self.h_sum += h
            self.h_count += 1
            self.h_final = h
        self._add(autonomy, "autonomy_sum", "autonomy_n", self)
        self._add(identity, "identity_sum", "identity_n", self)
        self._add(causal, "causal_sum", "causal_n", self)
        self._add(abs(agency) if math.isfinite(agency) else agency, "agency_sum", "agency_n", self)
        self._add(abs(memory) if math.isfinite(memory) else memory, "memory_sum", "memory_n", self)
        self._add(prediction_error, "pred_sum", "pred_n", self)
        self._add(terminal, "terminal_sum", "terminal_n", self)
        self._add(motor_alignment, "motor_sum", "motor_n", self)

    @staticmethod
    def _mean(total: float, n: int) -> float:
        return float(total / n) if n else float("nan")

    def summary(self) -> Dict[str, float]:
        return {
            "probe_opportunity_count": float(self.n),
            "probe_count": float(self.probes),
            "correct_probe_count": float(self.correct),
            "incorrect_probe_count": float(self.incorrect),
            "correct_probe_rate": float(self.correct / self.probes) if self.probes else float("nan"),
            "incorrect_probe_rate": float(self.incorrect / self.probes) if self.probes else float("nan"),
            "probe_balance": float((self.plus - self.minus) / self.probes) if self.probes else float("nan"),
            "cumulative_regret": float(self.incorrect),
            "direction_readout_accuracy": float(self.direction_correct / self.direction_available) if self.direction_available else float("nan"),
            "direction_readout_coverage": float(self.direction_available / self.n) if self.n else float("nan"),
            "direction_readout_abs_mean": float(self.direction_abs_sum / self.direction_available) if self.direction_available else float("nan"),
            "causal_direction_accuracy": float(self.causal_dir_correct / self.causal_dir_available) if self.causal_dir_available else float("nan"),
            "causal_direction_coverage": float(self.causal_dir_available / self.n) if self.n else float("nan"),
            "causal_direction_abs_mean": float(self.causal_dir_abs_sum / self.causal_dir_available) if self.causal_dir_available else float("nan"),
            "probe_readout_agreement_rate": float(self.direction_probe_agree / self.direction_probe_n) if self.direction_probe_n else float("nan"),
            "sensory_sign_accuracy": float(self.sensory_correct / self.sensory_available) if self.sensory_available else float("nan"),
            "sensory_sign_coverage": float(self.sensory_available / self.n) if self.n else float("nan"),
            "outcome_sum": float(self.outcome_sum),
            "damage_sum": float(self.damage_sum),
            "benefit_sum": float(self.benefit_sum),
            "h_auc": self._mean(self.h_sum, self.h_count),
            "h_final": float(self.h_final),
            "autonomy_mean": self._mean(self.autonomy_sum, self.autonomy_n),
            "identity_mean": self._mean(self.identity_sum, self.identity_n),
            "causal_confidence_mean": self._mean(self.causal_sum, self.causal_n),
            "agency_abs_mean": self._mean(self.agency_sum, self.agency_n),
            "memory_force_mean": self._mean(self.memory_sum, self.memory_n),
            "prediction_error_mean": self._mean(self.pred_sum, self.pred_n),
            "terminal_fraction": self._mean(self.terminal_sum, self.terminal_n),
            "motor_alignment_rate": self._mean(self.motor_sum, self.motor_n),
        }


def native_agent(mod: Any, regime: Regime, condition: Any, seed: int) -> Any:
    Agent = getattr(mod, "Agent")
    return Agent(make_params(mod, regime), copy.deepcopy(condition), int(seed))


def patch_integrated_regime(agent: Any, regime: Regime) -> None:
    if not hasattr(agent, "darca") or not hasattr(agent.darca, "agent"):
        raise RuntimeError("Integrated agent does not expose darca.agent")

    def regime_step(wrapper: Any, scalar_y: float, external_shock: float) -> Dict[str, Any]:
        env = {
            "external_shock": clip(float(external_shock), 0.0, 1.0),
            "d_dyn": float(regime.delay),
            "coupling_t": float(regime.coupling),
            "sigma_t": float(regime.noise),
        }
        return dict(wrapper.agent.step(float(scalar_y), env))

    agent.darca.step = types.MethodType(regime_step, agent.darca)


def integrated_outer_template(core: Any, darca_file: str, seed: int) -> Any:
    Agent = getattr(core, "IntegratedDARCAAgent")
    return Agent(darca_file=darca_file, seed=int(seed))


def integrated_condition_agent(
    template: Any,
    mod: Any,
    regime: Regime,
    condition: Any,
    seed: int,
) -> Any:
    agent = clone_without_module(template)
    agent.darca.module = mod
    agent.darca.agent = native_agent(mod, regime, condition, seed)
    patch_integrated_regime(agent, regime)
    return agent


def run_native_sequence(
    mod: Any,
    agent: Any,
    regime: Regime,
    noise: np.ndarray,
    rules: np.ndarray,
    cue_visible: bool,
    feedback_enabled: bool,
    outcome_magnitude: float,
    null_outcomes: Optional[np.ndarray] = None,
    split_step: Optional[int] = None,
) -> Tuple[List[MetricAccumulator], int]:
    n_steps = int(rules.size)
    pending = np.zeros(n_steps + int(regime.delay) + 1, dtype=np.float64)
    accumulators = [MetricAccumulator()] if split_step is None else [MetricAccumulator(), MetricAccumulator()]
    env = env_info_template(regime)
    calls = 0
    for t in range(n_steps):
        rule = int(rules[t])
        delivered = float(pending[t])
        cue = 0.30 * rule if cue_visible else 0.0
        y = clip(cue + delivered + float(noise[t]), -2.0, 2.0)
        env["external_shock"] = clip(abs(delivered), 0.0, 1.0)
        row = agent.step(y, env)
        calls += 1
        if not isinstance(row, Mapping):
            raise TypeError(f"Native Agent.step returned {type(row)}, expected Mapping")
        name = action_name(row, mod)
        sign = action_sign(name)
        if feedback_enabled and sign != 0:
            if null_outcomes is not None:
                outcome = float(null_outcomes[t]) * outcome_magnitude
            else:
                outcome = delayed_outcome_for_action(rule, sign, outcome_magnitude)
            due = t + int(regime.delay)
            if due < n_steps:
                pending[due] += outcome
        h = first_value(row, ("h", "life_h"))
        autonomy = first_value(row, ("autonomy", "a_t", "autonomy_state"))
        identity = first_value(row, ("identity", "identity_state"))
        causal = first_value(row, ("causal_confidence", "causal_conf", "causal_score"))
        agency = first_value(row, ("agency", "agency_state", "agency_force"))
        memory = first_value(row, ("memory_force", "memory", "memory_state"))
        pred = first_value(row, ("prediction_error", "pred_error", "causal_prediction_error"))
        terminal = first_value(row, ("terminal", "done", "is_terminal"), 0.0)
        direction_score = first_value(row, ("direction_readout_score", "intervention_direction"))
        causal_direction_score = first_value(row, ("causal_direction_score", "intervention_direction"))
        sensory_score = first_value(row, ("s", "e"))
        idx = 0 if split_step is None or t < int(split_step) else 1
        accumulators[idx].update(
            rule,
            sign,
            delivered,
            h,
            autonomy,
            identity,
            causal,
            agency,
            memory,
            pred,
            terminal,
            direction_score=direction_score,
            causal_direction_score=causal_direction_score,
            sensory_score=sensory_score,
        )
    return accumulators, calls


def run_integrated_sequence(
    core: Any,
    agent: Any,
    regime: Regime,
    noise: np.ndarray,
    rules: np.ndarray,
    cue_visible: bool,
    feedback_enabled: bool,
    outcome_magnitude: float,
    start_step: int = 0,
    null_outcomes: Optional[np.ndarray] = None,
    split_step: Optional[int] = None,
) -> Tuple[List[MetricAccumulator], int]:
    Obs = getattr(core, "AgentObservation")
    n_steps = int(rules.size)
    pending = np.zeros(n_steps + int(regime.delay) + 1, dtype=np.float64)
    accumulators = [MetricAccumulator()] if split_step is None else [MetricAccumulator(), MetricAccumulator()]
    obs = Obs(step=int(start_step), scalar_y=0.0)
    calls = 0
    for local_t in range(n_steps):
        rule = int(rules[local_t])
        delivered = float(pending[local_t])
        cue = 0.30 * rule if cue_visible else 0.0
        scalar_y = clip(cue + delivered + float(noise[local_t]), -2.0, 2.0)
        obs.step = int(start_step + local_t)
        obs.scalar_y = scalar_y
        obs.damage = max(0.0, -delivered)
        obs.resource_gain = max(0.0, delivered)
        obs.recovery_gain = max(0.0, delivered)
        obs.danger_pressure = clip(max(0.0, -delivered) * 3.0, 0.0, 1.0)
        obs.resource_pressure = clip(max(0.0, delivered) * 3.0, 0.0, 1.0)
        obs.novelty = 0.15
        obs.target_hint_dx = float(rule if cue_visible else 0.0)
        obs.target_hint_dy = 0.0
        obs.heading_x = 1.0
        obs.heading_y = 0.0
        obs.grounded = 1.0
        action = agent.step(obs)
        calls += 1
        name = str(getattr(action, "darca_action_name", "NA"))
        sign = action_sign(name)
        if feedback_enabled and sign != 0:
            if null_outcomes is not None:
                outcome = float(null_outcomes[local_t]) * outcome_magnitude
            else:
                outcome = delayed_outcome_for_action(rule, sign, outcome_magnitude)
            due = local_t + int(regime.delay)
            if due < n_steps:
                pending[due] += outcome
        motor_x = safe_float(getattr(action, "desired_vx", 0.0), 0.0)
        motor_alignment = float(np.sign(motor_x) == rule) if abs(motor_x) > 1e-12 else 0.0
        h = action_value(action, ("life_h", "h"))
        autonomy = action_value(action, ("autonomy", "a_t", "autonomy_state"))
        identity = action_value(action, ("identity", "identity_state"))
        causal = action_value(action, ("causal_confidence", "causal_conf", "causal_score"))
        agency = action_value(action, ("agency", "agency_state", "agency_force"))
        memory = action_value(action, ("memory_force", "memory", "memory_state"))
        pred = action_value(action, ("prediction_error", "pred_error", "physics_pred_error"))
        terminal = action_value(action, ("terminal", "done", "is_terminal"), 0.0)
        direction_score = action_value(action, ("direction_readout_score", "intervention_direction"))
        causal_direction_score = action_value(action, ("causal_direction_score", "intervention_direction"))
        sensory_score = action_value(action, ("s", "scalar_y"))
        idx = 0 if split_step is None or local_t < int(split_step) else 1
        accumulators[idx].update(
            rule,
            sign,
            delivered,
            h,
            autonomy,
            identity,
            causal,
            agency,
            memory,
            pred,
            terminal,
            motor_alignment,
            direction_score=direction_score,
            causal_direction_score=causal_direction_score,
            sensory_score=sensory_score,
        )
    return accumulators, calls


def constant_rules(rule: int, n: int) -> np.ndarray:
    return np.full(int(n), int(rule), dtype=np.int8)


def rule_for(base_seed: int, seed_id: int, task: str) -> int:
    # Exact seed-level counterbalancing for every task when the seed count is even.
    offset = stable_int(base_seed, task, "latent_rule_counterbalance") % 2
    return 1 if (int(seed_id) + int(offset)) % 2 == 0 else -1


def make_schedules(bundle: Bundle) -> Dict[str, Dict[str, np.ndarray]]:
    schedules: Dict[str, Dict[str, np.ndarray]] = {}
    for task in TASKS:
        schedules[task] = {}
    schedules["current_sufficient"]["noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "current_sufficient", bundle.regime, "full", bundle.steps
    )
    schedules["hidden_reversal"]["train_noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "hidden_reversal", bundle.regime, "train", bundle.history_steps
    )
    schedules["hidden_reversal"]["noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "hidden_reversal", bundle.regime, "full", bundle.steps
    )
    schedules["action_independent_null"]["noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "action_independent_null", bundle.regime, "full", bundle.steps
    )
    schedules["action_independent_null"]["null"] = schedule_null_outcomes(
        bundle.base_seed, bundle.seed_id, bundle.regime, bundle.steps
    )
    schedules["history_required"]["train_noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "history_required", bundle.regime, "train", bundle.history_steps
    )
    schedules["history_required"]["eval_noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "history_required", bundle.regime, "eval", bundle.eval_steps
    )
    baseline = Regime(4, 0.035, 0.45)
    schedules["transfer"]["train_noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "transfer", baseline, "train", bundle.history_steps
    )
    schedules["transfer"]["eval_noise"] = schedule_noise(
        bundle.base_seed, bundle.seed_id, "transfer", bundle.regime, "eval", bundle.eval_steps
    )
    return schedules


def add_common_metadata(
    row: Dict[str, Any],
    bundle: Bundle,
    task: str,
    condition: str,
    phase: str,
    initial_hash: str,
    model_seed: int,
) -> Dict[str, Any]:
    return {
        "task": task,
        "condition": condition,
        "seed_id": bundle.seed_id,
        "model_seed": model_seed,
        "regime_id": bundle.regime.id,
        "delay": bundle.regime.delay,
        "noise": bundle.regime.noise,
        "coupling": bundle.regime.coupling,
        "phase": phase,
        "initial_state_hash": initial_hash,
        **row,
    }


def execute_native_task(
    mod: Any,
    base: Any,
    bundle: Bundle,
    task: str,
    schedules: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, Any]], int]:
    magnitude = 0.18
    rows: List[Dict[str, Any]] = []
    calls = 0
    if task == "history_required":
        for history_rule in (-1, 1):
            trained = clone_without_module(base)
            train_rules = constant_rules(history_rule, bundle.history_steps)
            _, c = run_native_sequence(
                mod,
                trained,
                bundle.regime,
                schedules[task]["train_noise"],
                train_rules,
                False,
                True,
                magnitude,
            )
            calls += c
            for future_rule in (-1, 1):
                branch = clone_without_module(trained)
                eval_rules = constant_rules(future_rule, bundle.eval_steps)
                acc, c = run_native_sequence(
                    mod,
                    branch,
                    bundle.regime,
                    schedules[task]["eval_noise"],
                    eval_rules,
                    False,
                    False,
                    magnitude,
                )
                calls += c
                rows.append({
                    "history_rule": history_rule,
                    "future_rule": future_rule,
                    "history_matched": int(history_rule == future_rule),
                    **acc[0].summary(),
                })
        return rows, calls

    agent = clone_without_module(base)
    latent = rule_for(bundle.base_seed, bundle.seed_id, task)
    if task == "current_sufficient":
        rules = constant_rules(latent, bundle.steps)
        acc, calls = run_native_sequence(
            mod, agent, bundle.regime, schedules[task]["noise"], rules, True, True, magnitude
        )
        rows.append({"history_rule": 0, "future_rule": latent, "history_matched": -1, **acc[0].summary()})
        return rows, calls
    if task == "hidden_reversal":
        acquisition_rules = constant_rules(latent, bundle.history_steps)
        _, acquisition_calls = run_native_sequence(
            mod,
            agent,
            bundle.regime,
            schedules[task]["train_noise"],
            acquisition_rules,
            False,
            True,
            magnitude,
        )
        rules = np.empty(bundle.steps, dtype=np.int8)
        rules[: bundle.reversal_step] = latent
        rules[bundle.reversal_step :] = -latent
        acc, evaluation_calls = run_native_sequence(
            mod,
            agent,
            bundle.regime,
            schedules[task]["noise"],
            rules,
            False,
            True,
            magnitude,
            split_step=bundle.reversal_step,
        )
        calls = acquisition_calls + evaluation_calls
        pre, post = acc[0].summary(), acc[1].summary()
        row: Dict[str, Any] = {
            "history_rule": latent,
            "future_rule": -latent,
            "history_matched": 0,
            **post,
        }
        row.update({f"pre_{k}": v for k, v in pre.items()})
        row.update({f"post_{k}": v for k, v in post.items()})
        rows.append(row)
        return rows, calls
    if task == "transfer":
        baseline = Regime(4, 0.035, 0.45)
        train_rules = constant_rules(latent, bundle.history_steps)
        _, c1 = run_native_sequence(
            mod,
            agent,
            baseline,
            schedules[task]["train_noise"],
            train_rules,
            False,
            True,
            magnitude,
        )
        eval_rules = constant_rules(latent, bundle.eval_steps)
        acc, c2 = run_native_sequence(
            mod,
            agent,
            bundle.regime,
            schedules[task]["eval_noise"],
            eval_rules,
            False,
            False,
            magnitude,
        )
        rows.append({"history_rule": latent, "future_rule": latent, "history_matched": 1, **acc[0].summary()})
        return rows, c1 + c2
    if task == "action_independent_null":
        rules = constant_rules(latent, bundle.steps)
        acc, calls = run_native_sequence(
            mod,
            agent,
            bundle.regime,
            schedules[task]["noise"],
            rules,
            False,
            True,
            magnitude,
            null_outcomes=schedules[task]["null"],
        )
        rows.append({"history_rule": 0, "future_rule": latent, "history_matched": -1, **acc[0].summary()})
        return rows, calls
    raise ValueError(task)


def execute_integrated_task(
    core: Any,
    base: Any,
    bundle: Bundle,
    task: str,
    schedules: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, Any]], int]:
    magnitude = 0.18
    rows: List[Dict[str, Any]] = []
    calls = 0
    if task == "history_required":
        for history_rule in (-1, 1):
            trained = clone_without_module(base)
            train_rules = constant_rules(history_rule, bundle.history_steps)
            _, c = run_integrated_sequence(
                core,
                trained,
                bundle.regime,
                schedules[task]["train_noise"],
                train_rules,
                False,
                True,
                magnitude,
            )
            calls += c
            for future_rule in (-1, 1):
                branch = clone_without_module(trained)
                eval_rules = constant_rules(future_rule, bundle.eval_steps)
                acc, c = run_integrated_sequence(
                    core,
                    branch,
                    bundle.regime,
                    schedules[task]["eval_noise"],
                    eval_rules,
                    False,
                    False,
                    magnitude,
                    start_step=bundle.history_steps,
                )
                calls += c
                rows.append({
                    "history_rule": history_rule,
                    "future_rule": future_rule,
                    "history_matched": int(history_rule == future_rule),
                    **acc[0].summary(),
                })
        return rows, calls

    agent = clone_without_module(base)
    latent = rule_for(bundle.base_seed, bundle.seed_id, task)
    if task == "current_sufficient":
        rules = constant_rules(latent, bundle.steps)
        acc, calls = run_integrated_sequence(
            core, agent, bundle.regime, schedules[task]["noise"], rules, True, True, magnitude
        )
        rows.append({"history_rule": 0, "future_rule": latent, "history_matched": -1, **acc[0].summary()})
        return rows, calls
    if task == "hidden_reversal":
        acquisition_rules = constant_rules(latent, bundle.history_steps)
        _, acquisition_calls = run_integrated_sequence(
            core,
            agent,
            bundle.regime,
            schedules[task]["train_noise"],
            acquisition_rules,
            False,
            True,
            magnitude,
        )
        rules = np.empty(bundle.steps, dtype=np.int8)
        rules[: bundle.reversal_step] = latent
        rules[bundle.reversal_step :] = -latent
        acc, evaluation_calls = run_integrated_sequence(
            core,
            agent,
            bundle.regime,
            schedules[task]["noise"],
            rules,
            False,
            True,
            magnitude,
            start_step=bundle.history_steps,
            split_step=bundle.reversal_step,
        )
        calls = acquisition_calls + evaluation_calls
        pre, post = acc[0].summary(), acc[1].summary()
        row: Dict[str, Any] = {
            "history_rule": latent,
            "future_rule": -latent,
            "history_matched": 0,
            **post,
        }
        row.update({f"pre_{k}": v for k, v in pre.items()})
        row.update({f"post_{k}": v for k, v in post.items()})
        rows.append(row)
        return rows, calls
    if task == "transfer":
        baseline = Regime(4, 0.035, 0.45)
        train_rules = constant_rules(latent, bundle.history_steps)
        patch_integrated_regime(agent, baseline)
        _, c1 = run_integrated_sequence(
            core,
            agent,
            baseline,
            schedules[task]["train_noise"],
            train_rules,
            False,
            True,
            magnitude,
        )
        patch_integrated_regime(agent, bundle.regime)
        eval_rules = constant_rules(latent, bundle.eval_steps)
        acc, c2 = run_integrated_sequence(
            core,
            agent,
            bundle.regime,
            schedules[task]["eval_noise"],
            eval_rules,
            False,
            False,
            magnitude,
            start_step=bundle.history_steps,
        )
        rows.append({"history_rule": latent, "future_rule": latent, "history_matched": 1, **acc[0].summary()})
        return rows, c1 + c2
    if task == "action_independent_null":
        rules = constant_rules(latent, bundle.steps)
        acc, calls = run_integrated_sequence(
            core,
            agent,
            bundle.regime,
            schedules[task]["noise"],
            rules,
            False,
            True,
            magnitude,
            null_outcomes=schedules[task]["null"],
        )
        rows.append({"history_rule": 0, "future_rule": latent, "history_matched": -1, **acc[0].summary()})
        return rows, calls
    raise ValueError(task)


def run_bundle(bundle: Bundle) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        worker_init(bundle.darca_file, bundle.agent_core)
        mod = _WORKER_DARCA
        core = _WORKER_CORE
        conditions = _WORKER_CONDITIONS
        model_seed = int(bundle.base_seed + 1009 * bundle.seed_id)
        schedules = make_schedules(bundle)

        native_bases: Dict[str, Any] = {}
        native_hashes: Dict[str, str] = {}
        if bundle.run_native:
            native_bases = {
                name: native_agent(mod, bundle.regime, conditions[name], model_seed)
                for name in CONDITION_NAMES
            }
            native_hashes = {name: state_hash(agent) for name, agent in native_bases.items()}

        integrated_bases: Dict[str, Any] = {}
        integrated_hashes: Dict[str, str] = {}
        if bundle.run_integrated:
            template = integrated_outer_template(core, bundle.darca_file, model_seed)
            template.darca.module = mod
            for name in CONDITION_NAMES:
                agent = integrated_condition_agent(
                    template, mod, bundle.regime, conditions[name], model_seed
                )
                integrated_bases[name] = agent
                integrated_hashes[name] = state_hash(agent)

        rows: List[Dict[str, Any]] = []
        native_calls = 0
        integrated_calls = 0

        for condition_name in CONDITION_NAMES:
            if bundle.run_native:
                native_base = native_bases[condition_name]
                for task in TASKS:
                    task_rows, calls = execute_native_task(
                        mod, native_base, bundle, task, schedules
                    )
                    native_calls += calls
                    for row in task_rows:
                        rows.append(
                            add_common_metadata(
                                row,
                                bundle,
                                task,
                                condition_name,
                                "native",
                                native_hashes[condition_name],
                                model_seed,
                            )
                        )

            if bundle.run_integrated:
                integrated_base = integrated_bases[condition_name]
                for task in TASKS:
                    task_rows, calls = execute_integrated_task(
                        core, integrated_base, bundle, task, schedules
                    )
                    integrated_calls += calls
                    for row in task_rows:
                        rows.append(
                            add_common_metadata(
                                row,
                                bundle,
                                task,
                                condition_name,
                                "integrated",
                                integrated_hashes[condition_name],
                                model_seed,
                            )
                        )

        return {
            "ok": True,
            "schema_version": SCHEMA_VERSION,
            "experiment_hash": bundle.experiment_hash,
            "seed_id": bundle.seed_id,
            "regime_id": bundle.regime.id,
            "rows": rows,
            "native_calls": native_calls,
            "integrated_calls": integrated_calls,
            "worker_seconds": float(time.perf_counter() - started),
            "worker_import_count": int(_WORKER_IMPORT_COUNT),
            "native_initial_hashes": native_hashes,
            "integrated_initial_hashes": integrated_hashes,
        }
    except Exception:
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "experiment_hash": bundle.experiment_hash,
            "seed_id": bundle.seed_id,
            "regime_id": bundle.regime.id,
            "error": traceback.format_exc(),
            "bundle": asdict(bundle),
            "worker_seconds": float(time.perf_counter() - started),
        }


def history_matched_effects(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row["task"] != "history_required":
            continue
        key = (row["phase"], row["condition"], row["seed_id"], row["regime_id"])
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for key, group in groups.items():
        matched = [r for r in group if int(r["history_matched"]) == 1]
        mismatched = [r for r in group if int(r["history_matched"]) == 0]
        if len(matched) != 2 or len(mismatched) != 2:
            continue
        for metric in PRIMARY_METRICS:
            a = np.asarray([safe_float(r.get(metric)) for r in matched], dtype=float)
            b = np.asarray([safe_float(r.get(metric)) for r in mismatched], dtype=float)
            if np.isfinite(a).any() and np.isfinite(b).any():
                output.append({
                    "phase": key[0],
                    "condition": key[1],
                    "seed_id": key[2],
                    "regime_id": key[3],
                    "task": "history_required",
                    "metric": metric,
                    "matched_mean": float(np.nanmean(a)),
                    "mismatched_mean": float(np.nanmean(b)),
                    "effect": float(np.nanmean(a) - np.nanmean(b)),
                })
    return output


def build_contrast_arrays(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    history = history_matched_effects(rows)
    observations: List[Dict[str, Any]] = []
    for row in rows:
        if row["task"] == "history_required":
            continue
        for metric in PRIMARY_METRICS:
            observations.append({
                "phase": row["phase"],
                "task": row["task"],
                "condition": row["condition"],
                "seed_id": int(row["seed_id"]),
                "regime_id": row["regime_id"],
                "metric": metric,
                "value": safe_float(row.get(metric)),
            })
    for row in history:
        observations.append({
            "phase": row["phase"],
            "task": row["task"],
            "condition": row["condition"],
            "seed_id": int(row["seed_id"]),
            "regime_id": row["regime_id"],
            "metric": f"matched_history_advantage__{row['metric']}",
            "value": safe_float(row["effect"]),
        })

    index = {
        (o["phase"], o["task"], o["condition"], o["seed_id"], o["regime_id"], o["metric"]): o["value"]
        for o in observations
    }
    combinations = sorted(
        {(o["phase"], o["task"], o["regime_id"], o["metric"]) for o in observations}
    )
    records: List[Dict[str, Any]] = []
    for phase, task, regime_id, metric in combinations:
        available_seeds = sorted(
            {
                int(o["seed_id"])
                for o in observations
                if o["phase"] == phase
                and o["task"] == task
                and o["regime_id"] == regime_id
                and o["metric"] == metric
            }
        )
        for comparator in CONDITION_NAMES[1:]:
            seed_ids: List[int] = []
            diffs: List[float] = []
            for seed_id in available_seeds:
                a = index.get((phase, task, "Full", seed_id, regime_id, metric), float("nan"))
                b = index.get((phase, task, comparator, seed_id, regime_id, metric), float("nan"))
                if math.isfinite(a) and math.isfinite(b):
                    seed_ids.append(seed_id)
                    diffs.append(a - b)
            if diffs:
                records.append({
                    "phase": phase,
                    "task": task,
                    "regime_id": regime_id,
                    "contrast": f"Full_minus_{comparator}",
                    "metric": metric,
                    "seed_ids": tuple(seed_ids),
                    "x": np.asarray(diffs, dtype=np.float64),
                })
    return records


def bootstrap_weights(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, n, size=(n_boot, n), dtype=np.int16 if n < 32767 else np.int32)
    weights = np.zeros((n_boot, n), dtype=np.float32)
    rows = np.repeat(np.arange(n_boot), n)
    np.add.at(weights, (rows, indices.ravel()), 1.0 / n)
    return weights


def sign_matrix(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    bits = rng.integers(0, 2, size=(n_perm, n), dtype=np.int8)
    return (bits * 2 - 1).astype(np.float32) / float(n)


def paired_contrasts_batched(
    rows: Sequence[Mapping[str, Any]],
    n_boot: int,
    n_perm: int,
    seed: int,
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    records = build_contrast_arrays(rows)
    grouped: Dict[Tuple[int, Tuple[int, ...]], List[Dict[str, Any]]] = {}
    for record in records:
        key = (int(record["x"].size), tuple(record["seed_ids"]))
        grouped.setdefault(key, []).append(record)

    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    results: List[Dict[str, Any]] = []
    for (n, _seed_ids), group in grouped.items():
        boot_w = bootstrap_weights(n, n_boot, rng)
        sign_w = sign_matrix(n, n_perm, rng)
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            x_matrix = np.vstack([record["x"] for record in chunk])
            means = np.mean(x_matrix, axis=1)
            if n > 1:
                sds = np.std(x_matrix, axis=1, ddof=1)
            else:
                sds = np.full(len(chunk), np.nan)
            boot_means = x_matrix @ boot_w.T
            ci_low = np.quantile(boot_means, 0.025, axis=1)
            ci_high = np.quantile(boot_means, 0.975, axis=1)
            perm_abs = np.abs(x_matrix @ sign_w.T)
            observed_abs = np.abs(means)[:, None]
            pvals = (np.sum(perm_abs >= observed_abs - 1e-15, axis=1) + 1.0) / (n_perm + 1.0)
            for i, record in enumerate(chunk):
                sd = float(sds[i])
                results.append({
                    "phase": record["phase"],
                    "task": record["task"],
                    "regime_id": record["regime_id"],
                    "contrast": record["contrast"],
                    "metric": record["metric"],
                    "n_pairs": int(n),
                    "mean_difference": float(means[i]),
                    "ci95_low": float(ci_low[i]),
                    "ci95_high": float(ci_high[i]),
                    "paired_dz": float(means[i] / sd) if math.isfinite(sd) and sd > 0 else float("nan"),
                    "sign_flip_p": float(pvals[i]),
                })
    qvals = bh_fdr([safe_float(row["sign_flip_p"]) for row in results])
    for row, q in zip(results, qvals):
        row["q_global"] = q
    return results


def one_sample_capability_tests(
    rows: Sequence[Mapping[str, Any]],
    n_boot: int,
    n_perm: int,
    rng_seed: int,
) -> List[Dict[str, Any]]:
    """Seed-level tests against chance, plus history matched-minus-mismatched tests.

    Regimes are averaged within seed before inference, so the independent unit is
    the model seed rather than the individual regime or probe event.
    """
    records: List[Dict[str, Any]] = []
    rate_specs = {
        "correct_probe_rate": 0.5,
        "direction_readout_accuracy": 0.5,
        "causal_direction_accuracy": 0.5,
        "sensory_sign_accuracy": 0.5,
    }

    grouped: Dict[Tuple[str, str, str, str, int], List[float]] = {}
    for row in rows:
        if row["task"] == "history_required":
            continue
        for metric, null in rate_specs.items():
            value = safe_float(row.get(metric))
            if math.isfinite(value):
                key = (str(row["phase"]), str(row["task"]), str(row["condition"]), metric, int(row["seed_id"]))
                grouped.setdefault(key, []).append(value)
        if row["task"] == "hidden_reversal":
            for metric in ("pre_correct_probe_rate", "post_correct_probe_rate", "pre_direction_readout_accuracy", "post_direction_readout_accuracy", "pre_causal_direction_accuracy", "post_causal_direction_accuracy"):
                value = safe_float(row.get(metric))
                if math.isfinite(value):
                    key = (str(row["phase"]), str(row["task"]), str(row["condition"]), metric, int(row["seed_id"]))
                    grouped.setdefault(key, []).append(value)

    seed_values: Dict[Tuple[str, str, str, str], List[float]] = {}
    for (phase, task, condition, metric, seed_id), values in grouped.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            seed_values.setdefault((phase, task, condition, metric), []).append(float(np.mean(arr)))

    history = history_matched_effects(rows)
    history_grouped: Dict[Tuple[str, str, str, int], List[float]] = {}
    for row in history:
        if row["metric"] not in rate_specs:
            continue
        value = safe_float(row.get("effect"))
        if math.isfinite(value):
            key = (str(row["phase"]), str(row["condition"]), str(row["metric"]), int(row["seed_id"]))
            history_grouped.setdefault(key, []).append(value)
    for (phase, condition, metric, seed_id), values in history_grouped.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            seed_values.setdefault((phase, "history_required", condition, f"matched_advantage__{metric}"), []).append(float(np.mean(arr)))

    rng = np.random.default_rng(int(rng_seed) & 0xFFFFFFFF)
    for (phase, task, condition, metric), values in sorted(seed_values.items()):
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        null = 0.0 if metric.startswith("matched_advantage__") else 0.5
        diffs = x - null
        mean_value = float(np.mean(x))
        mean_difference = float(np.mean(diffs))
        if x.size == 1:
            ci_low = ci_high = mean_value
            p = 1.0
        else:
            indices = rng.integers(0, x.size, size=(n_boot, x.size))
            boot = np.mean(x[indices], axis=1)
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975]).tolist()
            signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, x.size))
            perm = np.mean(signs * diffs[None, :], axis=1)
            p = float((1.0 + np.sum(np.abs(perm) >= abs(mean_difference))) / (n_perm + 1.0))
        sd = float(np.std(diffs, ddof=1)) if x.size > 1 else float("nan")
        dz = float(mean_difference / sd) if math.isfinite(sd) and sd > 0.0 else float("nan")
        records.append({
            "phase": phase,
            "task": task,
            "condition": condition,
            "metric": metric,
            "null_value": null,
            "n_seeds": int(x.size),
            "mean": mean_value,
            "mean_minus_null": mean_difference,
            "bootstrap_ci95_low": float(ci_low),
            "bootstrap_ci95_high": float(ci_high),
            "paired_dz_vs_null": dz,
            "sign_flip_p": p,
        })
    qvals = bh_fdr([safe_float(row["sign_flip_p"]) for row in records])
    for row, q in zip(records, qvals):
        row["q_global"] = q
    return records


def preflight(darca_path: Path, core_path: Path) -> Dict[str, Any]:
    pid = os.getpid()
    mod = import_module(f"darca_fast_preflight_{pid}", darca_path)
    missing = [name for name in ("Params", "Condition", "Agent") if not hasattr(mod, name)]
    if missing:
        raise RuntimeError(f"DARCA source missing required symbols: {missing}")
    expected_readout_fix = "signed_action_conditioned_causal_tie_break_v1"
    if getattr(mod, "READOUT_FIX_ID", None) != expected_readout_fix:
        raise RuntimeError(
            "DARCA source does not contain the required signed causal-action readout fix: "
            f"expected {expected_readout_fix!r}, observed {getattr(mod, 'READOUT_FIX_ID', None)!r}"
        )
    conditions = extract_conditions(mod)
    condition_audit = audit_official_condition_definitions(conditions)
    regime = Regime(4, 0.035, 0.45)
    native_hashes: Dict[str, str] = {}
    native_parameter_hashes: Dict[str, str] = {}
    for name in CONDITION_NAMES:
        first = native_agent(mod, regime, conditions[name], 12345)
        second = native_agent(mod, regime, conditions[name], 12345)
        h1 = state_hash(first)
        h2 = state_hash(second)
        if h1 != h2:
            raise RuntimeError(
                f"Condition {name} is not reproducibly initialized from the same seed: "
                f"{h1} != {h2}"
            )
        native_hashes[name] = h1
        native_parameter_hashes[name] = state_hash(getattr(first, "p", None))
        row = first.step(0.0, env_info_template(regime))
        if not isinstance(row, Mapping):
            raise TypeError(f"Agent.step must return Mapping, got {type(row)}")
        if "h" not in row and "life_h" not in row:
            raise RuntimeError("Native Agent.step output has no h/life_h field")
        for required in ("intervention_direction", "direction_readout_score", "direction_readout_enabled"):
            if required not in row:
                raise RuntimeError(f"Native Agent.step output has no {required} field")
    if len(set(native_parameter_hashes.values())) != 1:
        raise RuntimeError(
            f"Official conditions do not share the same Params object: {native_parameter_hashes}"
        )

    core = import_module(f"integrated_fast_preflight_{pid}", core_path)
    for symbol in ("IntegratedDARCAAgent", "AgentObservation"):
        if not hasattr(core, symbol):
            raise RuntimeError(f"Integrated core missing {symbol}")
    loader_patched = patch_core_loader(core, mod)
    template = integrated_outer_template(core, str(darca_path), 12345)
    integrated_hashes: Dict[str, str] = {}
    integrated = None
    action = None
    for name in CONDITION_NAMES:
        first = integrated_condition_agent(template, mod, regime, conditions[name], 12345)
        second = integrated_condition_agent(template, mod, regime, conditions[name], 12345)
        h1 = state_hash(first)
        h2 = state_hash(second)
        if h1 != h2:
            raise RuntimeError(
                f"Integrated condition {name} is not reproducibly initialized from the same seed: "
                f"{h1} != {h2}"
            )
        integrated_hashes[name] = h1
        probe_action = first.step(core.AgentObservation(step=0, scalar_y=0.0))
        if not hasattr(probe_action, "darca_action_name"):
            raise RuntimeError(f"Integrated action for {name} has no darca_action_name")
        if name == "Full":
            integrated = first
            action = probe_action
    if integrated is None or action is None:
        raise RuntimeError("Full integrated condition was not initialized")
    clone = clone_without_module(integrated)
    clone.step(core.AgentObservation(step=1, scalar_y=0.0))

    # Actual online-accumulator path, not only a syntax check.
    rules = constant_rules(1, 4)
    noise = np.zeros(4, dtype=float)
    native_probe = native_agent(mod, regime, conditions["Full"], 7)
    native_acc, native_calls = run_native_sequence(
        mod, native_probe, regime, noise, rules, True, True, 0.18
    )
    integrated_probe = integrated_condition_agent(template, mod, regime, conditions["Full"], 7)
    integrated_acc, integrated_calls = run_integrated_sequence(
        core, integrated_probe, regime, noise, rules, True, True, 0.18
    )
    if native_acc[0].summary()["probe_opportunity_count"] != 4.0:
        raise RuntimeError("Native accumulator preflight failed")
    if integrated_acc[0].summary()["probe_opportunity_count"] != 4.0:
        raise RuntimeError("Integrated accumulator preflight failed")

    return {
        "darca_sha256": sha256_file(darca_path),
        "readout_fix_id": getattr(mod, "READOUT_FIX_ID", None),
        "agent_core_sha256": sha256_file(core_path),
        "native_agent_class": type(native_agent(mod, regime, conditions["Full"], 1)).__name__,
        "integrated_agent_class": type(integrated).__name__,
        "integrated_action_class": type(action).__name__,
        "condition_names": list(conditions),
        "pairing_policy": "same_model_seed_plus_shared_exogenous_schedules",
        "condition_definition_audit": condition_audit,
        "native_initial_state_hashes_by_condition": native_hashes,
        "integrated_initial_state_hashes_by_condition": integrated_hashes,
        "native_parameter_hash": next(iter(native_parameter_hashes.values())),
        "same_condition_seed_reproducibility": True,
        "cross_condition_state_hash_equality_required": False,
        "core_loader_cache_patch": bool(loader_patched),
        "native_online_probe_calls": native_calls,
        "integrated_online_probe_calls": integrated_calls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--darca-file", type=Path, required=True)
    parser.add_argument("--agent-core", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plan", choices=("smoke", "quick", "main"), default="smoke")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--bootstraps", type=int)
    parser.add_argument("--permutations", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native-only", action="store_true")
    mode.add_argument("--integrated-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_workers(value: str, cap: int, n_bundles: int) -> int:
    if value == "auto":
        return max(1, min(n_bundles, cap, max(1, (os.cpu_count() or 2) - 1)))
    return max(1, min(n_bundles, cap, int(value)))


def valid_saved_result(result: Any, experiment_hash: str) -> bool:
    return bool(
        isinstance(result, Mapping)
        and result.get("ok") is True
        and int(result.get("schema_version", -1)) == SCHEMA_VERSION
        and result.get("experiment_hash") == experiment_hash
    )


def main() -> None:
    args = parse_args()
    args.darca_file = args.darca_file.expanduser().resolve()
    args.agent_core = args.agent_core.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    for path in (args.darca_file, args.agent_core):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.overwrite and args.outdir.exists():
        shutil.rmtree(args.outdir)
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}. Use --overwrite or --resume."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    plan = plan_defaults(args.plan)
    seeds = int(args.seeds if args.seeds is not None else plan.seeds)
    n_boot = int(args.bootstraps if args.bootstraps is not None else plan.bootstraps)
    n_perm = int(args.permutations if args.permutations is not None else plan.permutations)
    regimes = build_regimes(args.plan)
    audit = preflight(args.darca_file, args.agent_core)
    experiment = {
        "script": SCRIPT_NAME,
        "schema": SCHEMA_VERSION,
        "plan": args.plan,
        "darca_sha256": audit["darca_sha256"],
        "agent_core_sha256": audit["agent_core_sha256"],
        "seeds": seeds,
        "tasks": TASKS,
        "conditions": CONDITION_NAMES,
        "regimes": [asdict(r) for r in regimes],
        "base_seed": args.base_seed,
        "run_native": bool(not args.integrated_only),
        "run_integrated": bool(not args.native_only),
        "steps": plan.steps,
        "history_steps": plan.history_steps,
        "eval_steps": plan.eval_steps,
        "reversal_step": plan.reversal_step,
    }
    experiment_hash = canonical_hash(experiment)
    audit.update({
        "created": now(),
        "script": SCRIPT_NAME,
        "schema_version": SCHEMA_VERSION,
        "darca_path": str(args.darca_file),
        "agent_core_path": str(args.agent_core),
        "experiment_hash": experiment_hash,
        "bundle_unit": "seed_x_regime",
        "pairing_policy": "same_model_seed_plus_shared_exogenous_schedules",
        "cross_condition_complete_state_hash_equality_required": False,
        "worker_import_policy": "once_per_worker",
        "per_step_rng_reconstruction": False,
        "step_dictionary_storage": False,
        "statistics_engine": "batched_vectorized",
    })
    (args.outdir / "01_MODEL_USE_AND_PREFLIGHT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    bundles = [
        Bundle(
            str(args.darca_file),
            str(args.agent_core),
            seed_id,
            args.base_seed,
            regime,
            plan.steps,
            plan.history_steps,
            plan.eval_steps,
            plan.reversal_step,
            not args.integrated_only,
            not args.native_only,
            experiment_hash,
        )
        for seed_id in range(seeds)
        for regime in regimes
    ]
    bundle_dir = args.outdir / "bundles"
    bundle_dir.mkdir(exist_ok=True)

    def bundle_path(bundle: Bundle) -> Path:
        return bundle_dir / f"seed{bundle.seed_id:03d}__{bundle.regime.id}.pkl"

    pending: List[Bundle] = []
    results: List[Dict[str, Any]] = []
    for bundle in bundles:
        path = bundle_path(bundle)
        if args.resume and path.is_file():
            try:
                with path.open("rb") as f:
                    saved = pickle.load(f)
                if valid_saved_result(saved, experiment_hash):
                    results.append(saved)
                    continue
            except Exception:
                pass
        pending.append(bundle)

    workers = resolve_workers(args.workers, plan.workers_cap, max(1, len(pending)))
    probe_seconds = float("nan")
    projected_seconds = float("nan")
    if pending:
        worker_init(str(args.darca_file), str(args.agent_core))
        probe_start = time.perf_counter()
        probe = run_bundle(pending[0])
        probe_seconds = float(time.perf_counter() - probe_start)
        if not probe.get("ok"):
            write_csv(args.outdir / "07_validation_errors.csv", [probe])
            raise RuntimeError("Pre-pool bundle probe failed:\n" + str(probe.get("error")))
        with bundle_path(pending[0]).open("wb") as f:
            pickle.dump(probe, f, protocol=pickle.HIGHEST_PROTOCOL)
        results.append(probe)
        pending = pending[1:]
        projected_seconds = probe_seconds * len(pending) / max(1, workers)
        print(
            f"[probe] bundle_seconds={probe_seconds:.2f} remaining={len(pending)} "
            f"workers={workers} projected_wall_seconds={projected_seconds:.1f}",
            flush=True,
        )

    wall_start = time.perf_counter()
    errors: List[Dict[str, Any]] = []
    print(
        f"[0.00s] bundles={len(bundles)} pending={len(pending)} workers={workers} "
        f"legacy_job_equivalent={len(bundles) * len(TASKS) * len(CONDITION_NAMES)}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(str(args.darca_file), str(args.agent_core)),
        ) as executor:
            future_map = {executor.submit(run_bundle, bundle): bundle for bundle in pending}
            completed = 0
            for future in as_completed(future_map):
                bundle = future_map[future]
                try:
                    result = future.result()
                except Exception:
                    result = {
                        "ok": False,
                        "schema_version": SCHEMA_VERSION,
                        "experiment_hash": experiment_hash,
                        "seed_id": bundle.seed_id,
                        "regime_id": bundle.regime.id,
                        "error": traceback.format_exc(),
                        "bundle": asdict(bundle),
                    }
                if result.get("ok"):
                    with bundle_path(bundle).open("wb") as f:
                        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                    results.append(result)
                else:
                    errors.append(result)
                completed += 1
                if completed % max(1, len(pending) // 20) == 0 or completed == len(pending):
                    print(
                        f"[{time.perf_counter()-wall_start:8.2f}s] "
                        f"completed={completed}/{len(pending)} errors={len(errors)}",
                        flush=True,
                    )
    wall_seconds = float(time.perf_counter() - wall_start)
    if errors:
        write_csv(args.outdir / "07_validation_errors.csv", errors)
        raise RuntimeError(f"{len(errors)} bundle(s) failed; see 07_validation_errors.csv")

    rows: List[Dict[str, Any]] = []
    native_calls = 0
    integrated_calls = 0
    worker_seconds = 0.0
    import_counts: List[int] = []
    for result in results:
        rows.extend(result["rows"])
        native_calls += int(result.get("native_calls", 0))
        integrated_calls += int(result.get("integrated_calls", 0))
        worker_seconds += safe_float(result.get("worker_seconds"), 0.0)
        import_counts.append(int(result.get("worker_import_count", 0)))

    expected_bundle_keys = {(seed, regime.id) for seed in range(seeds) for regime in regimes}
    actual_bundle_keys = {(int(r["seed_id"]), str(r["regime_id"])) for r in results if r.get("ok")}
    validation_errors: List[Dict[str, Any]] = []
    missing_bundles = sorted(expected_bundle_keys - actual_bundle_keys)
    for seed_id, regime_id in missing_bundles:
        validation_errors.append({
            "error": "missing_bundle",
            "seed_id": seed_id,
            "regime_id": regime_id,
        })

    phase_count = int(not args.integrated_only) + int(not args.native_only)
    expected_rows_per_bundle = len(CONDITION_NAMES) * 8 * phase_count
    # Each selected phase has: current 1 + history 4 + reversal 1 + transfer 1 + null 1 = 8 rows.
    for seed_id, regime_id in actual_bundle_keys:
        count = sum(
            1
            for row in rows
            if int(row["seed_id"]) == seed_id and str(row["regime_id"]) == regime_id
        )
        if count != expected_rows_per_bundle:
            validation_errors.append({
                "error": "bundle_row_count_mismatch",
                "seed_id": seed_id,
                "regime_id": regime_id,
                "observed": count,
                "expected": expected_rows_per_bundle,
            })

    if validation_errors:
        write_csv(args.outdir / "07_validation_errors.csv", validation_errors)
        raise RuntimeError(f"Validation failed for {len(validation_errors)} item(s)")

    write_csv(args.outdir / "02_raw_capability_cells.csv", rows)
    history = history_matched_effects(rows)
    write_csv(args.outdir / "03_history_matched_effects_by_seed.csv", history)

    stats_start = time.perf_counter()
    contrasts = paired_contrasts_batched(
        rows,
        n_boot,
        n_perm,
        stable_int(experiment_hash, "statistics"),
    )
    stats_seconds = float(time.perf_counter() - stats_start)
    write_csv(args.outdir / "04_all_full_minus_lesion_contrasts.csv", contrasts)
    primary_tests = one_sample_capability_tests(
        rows,
        n_boot,
        n_perm,
        stable_int(experiment_hash, "one_sample_capability_tests"),
    )
    write_csv(args.outdir / "08_PRIMARY_CAPABILITY_TESTS.csv", primary_tests)

    coverage: List[Dict[str, Any]] = []
    keys = sorted(
        {(r["phase"], r["task"], r["condition"], r["regime_id"]) for r in rows}
    )
    for phase, task, condition, regime_id in keys:
        subset = [
            r
            for r in rows
            if r["phase"] == phase
            and r["task"] == task
            and r["condition"] == condition
            and r["regime_id"] == regime_id
        ]

        def finite_mean(values: Sequence[float]) -> float:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size else float("nan")

        coverage.append({
            "phase": phase,
            "task": task,
            "condition": condition,
            "regime_id": regime_id,
            "n_rows": len(subset),
            "n_seeds": len({int(r["seed_id"]) for r in subset}),
            "mean_probe_count": finite_mean([safe_float(r.get("probe_count")) for r in subset]),
            "mean_correct_probe_rate": finite_mean(
                [safe_float(r.get("correct_probe_rate")) for r in subset]
            ),
            "mean_h_auc": finite_mean([safe_float(r.get("h_auc")) for r in subset]),
            "mean_incorrect_probe_rate": finite_mean([safe_float(r.get("incorrect_probe_rate")) for r in subset]),
            "mean_direction_readout_accuracy": finite_mean([safe_float(r.get("direction_readout_accuracy")) for r in subset]),
            "mean_causal_direction_accuracy": finite_mean([safe_float(r.get("causal_direction_accuracy")) for r in subset]),
            "mean_probe_readout_agreement_rate": finite_mean([safe_float(r.get("probe_readout_agreement_rate")) for r in subset]),
            "mean_sensory_sign_accuracy": finite_mean([safe_float(r.get("sensory_sign_accuracy")) for r in subset]),
        })
    write_csv(args.outdir / "05_task_coverage_and_raw_summary.csv", coverage)
    write_csv(args.outdir / "07_validation_errors.csv", [])

    performance = {
        "created": now(),
        "script": SCRIPT_NAME,
        "schema_version": SCHEMA_VERSION,
        "experiment_hash": experiment_hash,
        "bundle_count": len(bundles),
        "legacy_job_equivalent": len(bundles) * len(TASKS) * len(CONDITION_NAMES),
        "workers": workers,
        "probe_bundle_seconds": probe_seconds,
        "projected_remaining_wall_seconds_from_probe": projected_seconds,
        "observed_parallel_wall_seconds_excluding_probe": wall_seconds,
        "summed_worker_seconds": worker_seconds,
        "statistics_seconds": stats_seconds,
        "native_step_calls": native_calls,
        "integrated_step_calls": integrated_calls,
        "total_step_calls": native_calls + integrated_calls,
        "per_step_rng_reconstruction": False,
        "step_dictionary_storage": False,
        "action_to_row_calls_by_runner": 0,
        "external_schedule_generation": "once_per_task_phase_per_seed_regime_bundle",
        "module_import_policy": "once_per_worker",
        "bundle_unit": "seed_x_regime",
        "pairing_policy": "same_model_seed_plus_shared_exogenous_schedules",
        "cross_condition_complete_state_hash_equality_required": False,
        "maximum_worker_import_count_observed": max(import_counts) if import_counts else 0,
    }
    (args.outdir / "06_PERFORMANCE_AUDIT.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = args.outdir / "00_FIRST_READ_DARCA_v24_readout_corrected_intelligence_report.txt"
    report.write_text(
        "DARCA V24 READOUT-CORRECTED INTELLIGENCE-EMERGENCE BATTERY\n"
        + "=" * 92
        + "\n"
        + f"Created: {now()}\n"
        + f"Script: {SCRIPT_NAME}\n"
        + f"Experiment hash: {experiment_hash}\n"
        + f"DARCA source: {args.darca_file}\n"
        + f"DARCA SHA256: {audit['darca_sha256']}\n"
        + f"Readout fix: {audit.get('readout_fix_id')}\n"
        + f"Integrated core: {args.agent_core}\n"
        + f"Integrated core SHA256: {audit['agent_core_sha256']}\n"
        + f"Seeds: {seeds}\n"
        + f"Regimes: {len(regimes)}\n"
        + f"Bundles: {len(bundles)}\n"
        + f"Task-condition jobs represented: {len(bundles) * len(TASKS) * len(CONDITION_NAMES):,}\n"
        + f"Native Agent.step calls: {native_calls:,}\n"
        + f"IntegratedDARCAAgent.step calls: {integrated_calls:,}\n"
        + f"Probe bundle time: {probe_seconds:.2f} s\n"
        + f"Observed parallel wall time excluding probe: {wall_seconds:.2f} s\n"
        + f"Summed worker time: {worker_seconds:.2f} s\n"
        + f"Vectorized statistics time: {stats_seconds:.2f} s\n\n"
        + "Readout correction\n------------------\n"
        + "The original PROBE_PLUS and PROBE_MINUS utilities were exactly equal.\n"
        + "The corrected model preserves the original probe-class utility and uses the sign of\n"
        + "the existing action-conditioned causal prediction only to resolve PLUS versus MINUS.\n"
        + "Before a causal prediction exists, current signed sensory evidence resolves the tie.\n"
        + "Random_Readout retains the original random tie as a causal control.\n\n"
        + "Task validity\n-------------\n"
        + "history_required: acquisition feedback is present; evaluation feedback is absent.\n"
        + "transfer: acquisition occurs in the baseline regime; evaluation feedback is absent.\n"
        + "hidden_reversal: a separate acquisition phase precedes the scored pre/post reversal phases.\n"
        + "action_independent_null: outcomes remain independent of action.\n"
        + "Regimes are averaged within seed before one-sample inference.\n\n"
        + "Validation\n----------\n"
        + "PASS: required signed-readout source identifier verified.\n"
        + "PASS: exact supplied integrated core imported directly.\n"
        + "PASS: one worker import, continuous internal RNG, and shared external schedules.\n"
        + "PASS: Random_Readout differs from Full only by the declared readout switch.\n"
        + "PASS: each condition is deterministic for a fixed model seed and shares the same Params.\n"
        + "PASS: complete seed x regime coverage and expected row counts.\n"
        + "PASS: raw causal direction, effective readout direction, probe action, and sensory sign are exported separately.\n"
        + "PASS: cumulative regret is accompanied by incorrect_probe_rate; exposure count is not treated as accuracy.\n"
        + "PASS: no weighted intelligence score or result-dependent task selection.\n\n"
        + "Primary files\n-------------\n"
        + "02_raw_capability_cells.csv\n"
        + "03_history_matched_effects_by_seed.csv\n"
        + "04_all_full_minus_lesion_contrasts.csv\n"
        + "05_task_coverage_and_raw_summary.csv\n"
        + "06_PERFORMANCE_AUDIT.json\n"
        + "07_validation_errors.csv\n"
        + "08_PRIMARY_CAPABILITY_TESTS.csv\n",
        encoding="utf-8",
    )
    print(f"[done] {report}", flush=True)


if __name__ == "__main__":
    main()

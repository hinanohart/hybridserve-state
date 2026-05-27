"""A small, fully deterministic hybrid reference model used to *prove* the
rehydration-equivalence contract.

This is **not** a serving engine and not a useful language model — it is the
minimal artifact that exercises both halves of a hybrid stack with real carried
state, so that bitwise rehydration equivalence (and its adversarial negation) is
machine-checkable on CPU with no GPU and no heavyweight dependencies:

* a **gated-linear-attention-style** layer carrying a fixed-size
  ``recurrent_state`` (per-head ``d_k x d_v`` matrix) and a depthwise causal
  ``conv_state`` ring buffer, and
* a **softmax-attention** layer carrying a growing ``attn_kv`` cache.

Everything is ``float64`` NumPy with a fixed operation order, so two processes
that build the model from the same seed and feed the same tokens produce
**bitwise-identical** logits. Model weights are a pure function of the config
seed, so they need not be serialized — only the *state* is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import io as hss_io
from . import roles

KEY_NEXT_INPUT = "next_input"


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 32
    d_model: int = 16
    n_heads: int = 2
    conv_kernel: int = 3
    layer_types: tuple[str, ...] = ("recurrent", "attn", "recurrent")
    seed: int = 0

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    def to_metadata(self) -> dict[str, str]:
        return {
            roles.KEY_MODEL_ARCH: "hybridserve-reference",
            "cfg.vocab": str(self.vocab),
            "cfg.d_model": str(self.d_model),
            "cfg.n_heads": str(self.n_heads),
            "cfg.conv_kernel": str(self.conv_kernel),
            "cfg.layer_types": ",".join(self.layer_types),
            "cfg.seed": str(self.seed),
        }

    @classmethod
    def from_metadata(cls, m: dict[str, str]) -> "ModelConfig":
        return cls(
            vocab=int(m["cfg.vocab"]),
            d_model=int(m["cfg.d_model"]),
            n_heads=int(m["cfg.n_heads"]),
            conv_kernel=int(m["cfg.conv_kernel"]),
            layer_types=tuple(m["cfg.layer_types"].split(",")),
            seed=int(m["cfg.seed"]),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _build_weights(cfg: ModelConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    d, v, k = cfg.d_model, cfg.vocab, cfg.conv_kernel
    scale = 1.0 / np.sqrt(d)
    w: dict[str, np.ndarray] = {
        "embed": rng.standard_normal((v, d)) * 0.5,
        "lm_head": rng.standard_normal((d, v)) * scale,
    }
    for i, lt in enumerate(cfg.layer_types):
        if lt == "recurrent":
            w[f"{i}.conv"] = rng.standard_normal((d, k)) * 0.3
            for p in ("Wq", "Wk", "Wv", "Wg", "Wo"):
                w[f"{i}.{p}"] = rng.standard_normal((d, d)) * scale
        elif lt == "attn":
            for p in ("Wq", "Wk", "Wv", "Wo"):
                w[f"{i}.{p}"] = rng.standard_normal((d, d)) * scale
        else:
            raise ValueError(f"unknown layer type: {lt}")
    return w


@dataclass
class ReferenceModel:
    cfg: ModelConfig
    w: dict[str, np.ndarray] = field(init=False)
    layers: list[dict] = field(init=False)
    seen: int = field(init=False, default=0)
    next_input: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.w = _build_weights(self.cfg)
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        d, h, dh, k = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.conv_kernel
        self.seen = 0
        self.next_input = 0
        self.layers = []
        for lt in cfg.layer_types:
            if lt == "recurrent":
                self.layers.append(
                    {
                        "type": "recurrent",
                        "S": np.zeros((h, dh, dh), dtype=np.float64),
                        "conv_ring": np.zeros((d, k), dtype=np.float64),
                        "phase": 0,
                    }
                )
            else:
                self.layers.append(
                    {
                        "type": "attn",
                        "K": np.zeros((0, h, dh), dtype=np.float64),
                        "V": np.zeros((0, h, dh), dtype=np.float64),
                    }
                )

    def step(self, token: int) -> np.ndarray:
        """Consume one token, mutate all layer states, return logits."""
        cfg = self.cfg
        d, h, dh, k = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.conv_kernel
        x = self.w["embed"][token].copy()

        for i, lt in enumerate(cfg.layer_types):
            st = self.layers[i]
            if lt == "recurrent":
                ring, phase = st["conv_ring"], st["phase"]
                ring[:, phase] = x
                phase = (phase + 1) % k
                st["phase"] = phase
                order = [(phase + j) % k for j in range(k)]  # oldest -> newest
                window = ring[:, order]
                conv_out = np.sum(window * self.w[f"{i}.conv"], axis=1)
                hh = np.tanh(conv_out)

                q = (hh @ self.w[f"{i}.Wq"]).reshape(h, dh)
                kk = (hh @ self.w[f"{i}.Wk"]).reshape(h, dh)
                vv = (hh @ self.w[f"{i}.Wv"]).reshape(h, dh)
                g = _sigmoid(hh @ self.w[f"{i}.Wg"]).reshape(h, dh)

                s = st["S"]
                o = np.empty((h, dh), dtype=np.float64)
                for head in range(h):
                    s[head] = g[head][:, None] * s[head] + np.outer(kk[head], vv[head])
                    o[head] = q[head] @ s[head]
                out = o.reshape(d) @ self.w[f"{i}.Wo"]
                x = x + out
            else:  # attn
                q = (x @ self.w[f"{i}.Wq"]).reshape(h, dh)
                kk = (x @ self.w[f"{i}.Wk"]).reshape(h, dh)
                vv = (x @ self.w[f"{i}.Wv"]).reshape(h, dh)
                st["K"] = np.concatenate([st["K"], kk[None]], axis=0)
                st["V"] = np.concatenate([st["V"], vv[None]], axis=0)
                kc, vc = st["K"], st["V"]
                o = np.empty((h, dh), dtype=np.float64)
                for head in range(h):
                    scores = (kc[:, head, :] @ q[head]) / np.sqrt(dh)
                    o[head] = _softmax(scores) @ vc[:, head, :]
                out = o.reshape(d) @ self.w[f"{i}.Wo"]
                x = x + out

        self.seen += 1
        return x @ self.w["lm_head"]

    def warm_prompt(self, prompt: list[int]) -> None:
        """Feed all but the last prompt token; the last becomes ``next_input``."""
        if not prompt:
            raise ValueError("prompt must be non-empty")
        for t in prompt[:-1]:
            self.step(t)
        self.next_input = int(prompt[-1])

    def gen_steps(self, n: int) -> list[tuple[int, np.ndarray]]:
        """Greedily generate ``n`` tokens, returning ``(token, logits)`` pairs."""
        out: list[tuple[int, np.ndarray]] = []
        for _ in range(n):
            logits = self.step(self.next_input)
            tok = int(np.argmax(logits))
            out.append((tok, logits.copy()))
            self.next_input = tok
        return out

    def capture_state(self) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        state: dict[str, np.ndarray] = {}
        meta: dict[str, str] = self.cfg.to_metadata()
        meta[roles.KEY_N_LAYERS] = str(len(self.cfg.layer_types))
        meta[roles.KEY_SEEN_TOKENS] = str(self.seen)
        meta[KEY_NEXT_INPUT] = str(self.next_input)
        for i, lt in enumerate(self.cfg.layer_types):
            st = self.layers[i]
            if lt == "recurrent":
                state[f"layers.{i}.recurrent_state"] = st["S"]
                state[f"layers.{i}.conv_state"] = st["conv_ring"]
                meta[roles.layer_role_key(i)] = roles.RECURRENT_STATE
                meta[roles.layer_conv_phase_key(i)] = str(st["phase"])
            else:
                state[f"layers.{i}.attn_kv.k"] = st["K"]
                state[f"layers.{i}.attn_kv.v"] = st["V"]
                meta[roles.layer_role_key(i)] = roles.ATTN_KV
        return state, meta

    def load_state(self, state: dict[str, np.ndarray], meta: dict[str, str]) -> None:
        self.seen = int(meta[roles.KEY_SEEN_TOKENS])
        self.next_input = int(meta[KEY_NEXT_INPUT])
        for i, lt in enumerate(self.cfg.layer_types):
            st = self.layers[i]
            if lt == "recurrent":
                st["S"] = np.array(
                    state[f"layers.{i}.recurrent_state"], dtype=np.float64
                )
                st["conv_ring"] = np.array(
                    state[f"layers.{i}.conv_state"], dtype=np.float64
                )
                st["phase"] = int(meta[roles.layer_conv_phase_key(i)])
            else:
                st["K"] = np.array(state[f"layers.{i}.attn_kv.k"], dtype=np.float64)
                st["V"] = np.array(state[f"layers.{i}.attn_kv.v"], dtype=np.float64)

    def save_state(self, path: str) -> None:
        state, meta = self.capture_state()
        hss_io.save(path, state, meta)

    @classmethod
    def restore(cls, path: str) -> "ReferenceModel":
        state, meta = hss_io.load(path)
        model = cls(ModelConfig.from_metadata(meta))
        model.load_state(state, meta)
        return model

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

PromptFormat = Literal["kv", "json"]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return f"{float(v):.6g}"
    return str(v)


def _format_features(
    feats: dict[str, Any],
    *,
    prompt_format: PromptFormat,
    max_features: int | None,
) -> str:
    keys = sorted(feats.keys())
    if max_features is not None:
        keys = keys[: max(0, int(max_features))]

    if prompt_format == "json":
        sub = {k: feats.get(k) for k in keys}
        return json.dumps(sub, sort_keys=True)

    # kv
    lines: list[str] = []
    for k in keys:
        lines.append(f"{k}={_format_scalar(feats.get(k))}")
    return "\n".join(lines)


def _make_prompt(
    *,
    task: str,
    labels: list[str],
    features_text: str,
    params: dict[str, Any] | None,
    include_params: bool,
) -> str:
    parts: list[str] = []
    parts.append("You are given features extracted from a tetrakis-sim simulation run.")
    parts.append(f"Task: {task}")
    parts.append(f"Possible labels: {', '.join(labels)}.")
    parts.append("Reply with only one label (no extra text).")

    if include_params and params is not None:
        parts.append("")
        parts.append("Params:")
        # Keep params stable and readable
        parts.append(json.dumps(params, sort_keys=True))

    parts.append("")
    parts.append("Features:")
    parts.append(features_text)
    return "\n".join(parts)


def export_llm_eval_jsonl(
    data_path: str | Path,
    out_path: str | Path,
    *,
    prompt_format: PromptFormat = "kv",
    max_features: int | None = None,
    include_params: bool = False,
) -> Path:
    """Export a tetrakis-sim eval dataset JSONL into prompt/expected JSONL.

    This does NOT call any API and costs nothing by itself. It only writes a file.
    """
    rows = _load_jsonl(data_path)
    if not rows:
        raise ValueError("Dataset is empty")

    labels = sorted({str(r.get("label")) for r in rows if r.get("label") is not None})
    if not labels:
        raise ValueError("No labels found in dataset")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            task = str(r.get("task", "defect_classification"))
            expected = str(r.get("label"))
            feats_any = r.get("features") or {}
            if not isinstance(feats_any, dict):
                raise ValueError("Record features must be a dict")

            feats = feats_any
            params_any = r.get("params") or None
            params = params_any if isinstance(params_any, dict) else None

            features_text = _format_features(
                feats,
                prompt_format=prompt_format,
                max_features=max_features,
            )
            prompt = _make_prompt(
                task=task,
                labels=labels,
                features_text=features_text,
                params=params,
                include_params=include_params,
            )

            out_rec = {
                "id": r.get("id"),
                "task": task,
                "prompt": prompt,
                "expected": expected,
                "metadata": {
                    "schema_version": r.get("schema_version"),
                    "source_id": r.get("id"),
                },
            }
            f.write(json.dumps(out_rec, sort_keys=True) + "\n")

    return out_path

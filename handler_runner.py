"""Tiny launcher: invoke handler.handler with role from $ROLE.

Used by apptainer exec when no Lambda RIE is in the loop. Mirrors the
%runscript in unified.def but lives as a separate file so we can also
invoke it via `apptainer exec ... python /opt/unified/handler_runner.py`
on a sandbox that didn't get the .def's runscript baked in (e.g. one
built directly from docker://python:3.10-slim).
"""
from __future__ import annotations

import json
import os
import sys
import types

# handler.py lives in the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import handler  # noqa: E402  triggers eager prewarm of both roles

role = os.environ.get("ROLE", "actor")
event = {"role": role}
if role == "learner":
    event["batch_id"] = os.environ.get("BATCH_ID", "seed")

ctx = types.SimpleNamespace(aws_request_id="apptainer-exec")
result = handler.handler(event, ctx)
print(json.dumps(result, indent=2, default=str))

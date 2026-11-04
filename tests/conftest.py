"""Keep the offline tests offline: metrics would otherwise log traces to Opik."""

import os

os.environ["OPIK_TRACK_DISABLE"] = "true"

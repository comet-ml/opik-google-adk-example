import os

from dotenv import load_dotenv

load_dotenv()

# A blank OPIK_* line in .env would override `opik configure` (~/.opik.config)
# and break self-hosted setups, so treat blank as unset.
for _name in ("OPIK_API_KEY", "OPIK_WORKSPACE", "OPIK_URL_OVERRIDE"):
    if os.environ.get(_name, None) == "":
        del os.environ[_name]

from . import agent  # noqa: F401  (lets `adk web` and `adk run` find root_agent)

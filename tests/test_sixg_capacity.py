import math
import os
import sys
import importlib.util
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if importlib.util.find_spec("numpy") is None:
    pytest.skip("numpy required for environment module", allow_module_level=True)

from src.environments.multi_agent_traffic_env import (
    SixGNetwork,
    DEFAULT_6G_BANDWIDTH,
)


def test_capacity_no_path_loss():
    net = SixGNetwork(bandwidth=DEFAULT_6G_BANDWIDTH, snr=10.0, path_loss_db=0.0)
    expected = DEFAULT_6G_BANDWIDTH * math.log2(1 + 10.0)
    assert math.isclose(net.calculate_capacity(), expected, rel_tol=1e-6)


def test_capacity_with_path_loss():
    net = SixGNetwork(bandwidth=DEFAULT_6G_BANDWIDTH, snr=10.0, path_loss_db=10.0)
    effective_snr = 10.0 * (10 ** (-10.0 / 10))
    expected = DEFAULT_6G_BANDWIDTH * math.log2(1 + effective_snr)
    assert math.isclose(net.calculate_capacity(), expected, rel_tol=1e-6)

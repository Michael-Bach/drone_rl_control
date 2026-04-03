"""Tests for RadarReceiver — no hardware required (serial is mocked)."""
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture()
def fake_serial(monkeypatch):
    """Inject a fake `serial` module and patch the radar module's reference to it."""
    mod = MagicMock()
    mod.SerialException = Exception
    monkeypatch.setitem(sys.modules, "serial", mod)

    # Patch the already-imported radar module's serial reference and availability flag.
    import drone_rl.utils.radar as radar_mod
    monkeypatch.setattr(radar_mod, "serial", mod)
    monkeypatch.setattr(radar_mod, "_SERIAL_AVAILABLE", True)
    return mod


def _make_conn(fake_serial, lines):
    """Return a mock serial connection whose readline() returns *lines* in order."""
    conn = MagicMock()
    conn.readline.side_effect = list(lines) + [b""] * 500
    fake_serial.Serial.return_value = conn
    return conn


def test_latest_zeros_before_start():
    from drone_rl.utils.radar import RadarReceiver
    r = RadarReceiver(port="/dev/null")
    np.testing.assert_array_equal(r.latest, np.zeros(9, dtype=np.float32))


def test_latest_returns_independent_copy():
    from drone_rl.utils.radar import RadarReceiver
    r = RadarReceiver(port="/dev/null")
    a = r.latest
    b = r.latest
    assert a is not b


def test_start_without_serial_installed_raises(monkeypatch):
    import drone_rl.utils.radar as radar_mod
    monkeypatch.setattr(radar_mod, "_SERIAL_AVAILABLE", False)
    from drone_rl.utils.radar import RadarReceiver
    r = RadarReceiver(port="/dev/null")
    with pytest.raises(ImportError, match="pyserial"):
        r.start()


def test_valid_csv_line_updates_latest(fake_serial):
    _make_conn(fake_serial, [b"100,-200,15,300,-400,20,0,0,0\n"])
    from drone_rl.utils.radar import RadarReceiver
    r = RadarReceiver(port="/dev/null")
    r.start()
    time.sleep(0.05)
    r.stop()
    expected = np.array([100, -200, 15, 300, -400, 20, 0, 0, 0], dtype=np.float32)
    np.testing.assert_array_equal(r.latest, expected)


def test_malformed_line_skipped(fake_serial):
    _make_conn(fake_serial, [b"bad,data\n", b"1,2,3,4,5,6,7,8,9\n"])
    from drone_rl.utils.radar import RadarReceiver
    r = RadarReceiver(port="/dev/null")
    r.start()
    time.sleep(0.05)
    r.stop()
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    np.testing.assert_array_equal(r.latest, expected)


def test_non_numeric_line_skipped(fake_serial):
    _make_conn(fake_serial, [b"x,y,z,a,b,c,d,e,f\n", b"1,2,3,4,5,6,7,8,9\n"])
    from drone_rl.utils.radar import RadarReceiver
    r = RadarReceiver(port="/dev/null")
    r.start()
    time.sleep(0.05)
    r.stop()
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    np.testing.assert_array_equal(r.latest, expected)

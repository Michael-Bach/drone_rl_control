"""LD2450 Bluetooth SPP radar receiver."""
from __future__ import annotations

import sys
import threading
from typing import Optional

import numpy as np

try:
    import serial
    from serial import SerialException as _SerialException
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore[assignment]
    _SerialException = OSError  # fallback so except clause is always valid
    _SERIAL_AVAILABLE = False


class RadarReceiver:
    """
    Thread-safe reader for LD2450 CSV data streamed over Bluetooth SPP.

    The ESP32 emits one line per LD2450 frame:
        ``x1,y1,s1,x2,y2,s2,x3,y3,s3\\n``
    where x/y are in mm (int) and s is speed in cm/s (int).

    Usage::

        r = RadarReceiver(port="/dev/rfcomm0")
        r.start()
        obs = r.latest   # np.ndarray shape (9,), dtype float32
        r.stop()
    """

    def __init__(self, port: str, baud: int = 9600) -> None:
        self._port = port
        self._baud = baud
        self._latest = np.zeros(9, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._conn: Optional[object] = None

    def start(self) -> None:
        """Open the serial port and start the background reader thread."""
        if not _SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for RadarReceiver. "
                "Install with: pip install 'drone-rl[hardware]'"
            )
        self._conn = serial.Serial(self._port, self._baud, timeout=1.0)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    @property
    def latest(self) -> np.ndarray:
        """Return a copy of the most recently received radar frame (shape (9,))."""
        with self._lock:
            return self._latest.copy()

    def stop(self) -> None:
        """Signal the reader thread to stop and close the serial port."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._conn is not None:
            self._conn.close()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._conn.readline()
                if not raw:
                    continue
                parts = raw.decode("ascii", errors="ignore").strip().split(",")
                if len(parts) != 9:
                    continue
                values = np.array([float(p) for p in parts], dtype=np.float32)
            except (ValueError, _SerialException) as exc:
                print(f"radar_bt_receiver: {exc}", file=sys.stderr)
                continue
            with self._lock:
                self._latest = values

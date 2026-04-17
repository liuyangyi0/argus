"""External sensor fusion — generic (camera_id, zone_id) → multiplier signals.

Any external signal source (temperature, vibration, radiation, lidar, etc.)
can push a short-lived multiplier that biases the alert grader's severity
for a given camera/zone. The fusion layer is deliberately sensor-agnostic —
it stores tuples and does not care what they represent.
"""

from argus.sensors.fusion import SensorFusion

__all__ = ["SensorFusion"]

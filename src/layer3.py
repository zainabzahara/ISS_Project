import numpy as np
from collections import deque

class Layer3PhysicsValidator:
    def __init__(self, history_window=500):
        """
        Initializes the Physics-Based Consistency Engine.
        
        :param history_window: The number of past readings to establish the normal 
                               physical baseline for Motor Current and Oil Temperature.
        """
        self.history_window = history_window
        
        # Moving histories for physical correlation sensors
        self.motor_current_history = deque(maxlen=history_window)
        self.oil_temp_history = deque(maxlen=history_window)

    def update_baselines(self, data_row):
        """
        Continuously updates the normal physical operating baseline.
        This should be called in the main loop to keep the "memory" of the machine fresh.
        """
        self.motor_current_history.append(data_row['Motor_current'])
        self.oil_temp_history.append(data_row['Oil_temperature'])

    def classify_root_cause(self, data_row):
        """
        Called ONLY when Layer 2 detects a statistical anomaly in Pressure (TP2).
        Evaluates physical consistency to distinguish between FDI and a mechanical fault.
        """
        # If we don't have enough data to establish physical laws yet, default to warming up
        if len(self.motor_current_history) < 100:
            return "WARNING: Anomaly Detected (Warming up physical baselines...)"

        # 1. Calculate the current physical baselines
        mean_current = np.mean(self.motor_current_history)
        std_current = np.std(self.motor_current_history) + 1e-6
        
        mean_temp = np.mean(self.oil_temp_history)
        std_temp = np.std(self.oil_temp_history) + 1e-6

        # 2. Get the current values at the exact moment of the anomaly
        current_val = data_row['Motor_current']
        temp_val = data_row['Oil_temperature']

        # 3. Check if motor current or oil temperature deviate physically (using 3-sigma rule)
        # In a real air compressor, a major pressure issue causes the motor to strain or heat up
        current_is_anomalous = abs(current_val - mean_current) > 3 * std_current
        temp_is_anomalous = abs(temp_val - mean_temp) > 3 * std_temp

        # 4. The Decision Engine
        if current_is_anomalous or temp_is_anomalous:
            # The physical laws ARE consistent: a pressure issue is accompanied by motor strain
            return ">> ALARM: HARDWARE FAULT! (Mechanical issue: Air Leak / Compressor Strain)"
        else:
            # The physical laws are VIOLATED: pressure spiked, but the motor is perfectly fine
            return ">> ALARM: CYBER ATTACK (FDI)! (Sensor spoofing detected: Physical laws violated)"
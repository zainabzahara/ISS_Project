import numpy as np
from scipy.optimize import minimize
from collections import deque

class Layer2StatDetector:
    def __init__(self, time_window=1000, lambda_history_size=50):
        """
        Initializes the Statistical Anomaly Detector.
        
        :param time_window: The moving window size (in time steps) to look back for estimating lambda(t).
        :param lambda_history_size: How many past lambda estimates to keep for the 3-sigma rule.
        """
        self.time_window = time_window
        self.current_time = 0.0
        
        # Deques for efficient moving windows
        self.recent_arrivals = deque()  # Stores timestamps of detected spikes
        self.lambda_history = deque(maxlen=lambda_history_size) # Stores estimated lambda rates
        self.tp2_history = deque(maxlen=500) # Baseline normal TP2 readings

    def detect_arrival(self, tp2_value):
        """
        Step 1: Detects if the current data point is a physical anomaly (an "arrival" in our Poisson Process).
        Uses a rolling baseline to capture basic deviations before doing the heavy MLE math.
        """
        # Bootstrap phase: Fill the baseline history first
        if len(self.tp2_history) < 100:
            self.tp2_history.append(tp2_value)
            return False

        mean_tp2 = np.mean(self.tp2_history)
        std_tp2 = np.std(self.tp2_history) + 1e-6 # prevent zero division
        
        # If the value deviates significantly (e.g. 4 standard deviations), log it as an "arrival"
        if abs(tp2_value - mean_tp2) > 4 * std_tp2:
            return True
        else:
            self.tp2_history.append(tp2_value)
            return False

    def estimate_intensity_mle(self):
        """
        Step 2: The Core Math. Uses Maximum Likelihood Estimation to find the time-varying 
        intensity lambda(t) = a + b*t of the anomalous arrivals within the moving window.
        """
        if len(self.recent_arrivals) < 3:
            return 0.0 # Not enough anomalies to establish a statistical rate

        arrivals = np.array(self.recent_arrivals)
        T = self.time_window
        
        # Shift arrival times so the current window starts at t=0 and ends at t=T
        shifted_arrivals = arrivals - (self.current_time - T)

        # Define the Negative Log-Likelihood (NLL) function for lambda(t) = a + b*t
        # Math: l(lambda) = - [a*T + 0.5*b*T^2] + Sum( log(a + b*tau_i) )
        def neg_log_likelihood(params):
            a, b = params
            integral = a * T + 0.5 * b * (T**2) # The expected compensator A(T)
            
            rates = a + b * shifted_arrivals
            # Constraint: rate lambda(t) must be > 0. If optimizer tries a negative rate, heavily penalize it.
            if np.any(rates <= 0):
                return np.inf 
                
            sum_log = np.sum(np.log(rates))
            
            return integral - sum_log # We MINIMIZE the NEGATIVE log-likelihood

        # Initial guess: constant average rate over the window
        initial_rate = len(arrivals) / T
        
        # Optimizer bounds & constraints: 'a' must be > 0, and lambda at the end of window (a + b*T) must be > 0
        bounds = ((1e-4, None), (None, None))
        cons = ({'type': 'ineq', 'fun': lambda p: p + p[1]*T - 1e-4})

        # Run the SciPy SLSQP optimizer
        res = minimize(neg_log_likelihood, x0=[initial_rate, 0.0], bounds=bounds, constraints=cons, method='SLSQP')

        if res.success:
            a, b = res.x
            current_lambda = a + b * T # The instantaneous attack rate right NOW
            return current_lambda
        else:
            return initial_rate

    def check_statistical_alert(self, data_row, time_t):
        """
        Step 3: Orchestrates the layer. Called continuously in the main loop.
        Returns True if a Non-Stationary Poisson shift is detected.
        """
        self.current_time = time_t
        tp2_value = data_row['TP2']

        # 1. Log arrival times of anomalies
        is_arrival = self.detect_arrival(tp2_value)
        if is_arrival:
            self.recent_arrivals.append(time_t)

        # Remove old arrivals that fell outside our moving time window
        while self.recent_arrivals and self.recent_arrivals[0] < time_t - self.time_window:
            self.recent_arrivals.popleft()

        # 2. Calculate the instantaneous arrival rate lambda(t)
        current_lambda = self.estimate_intensity_mle()

        # 3. Apply the 3-Sigma Rule to detect if the rate is shifting maliciously
        alert_triggered = False
        
        if len(self.lambda_history) == self.lambda_history.maxlen:
            mean_lambda = np.mean(self.lambda_history)
            std_lambda = np.std(self.lambda_history) + 1e-6

            # If the calculated frequency jumps out of normal bounds -> Cyber Attack Burst!
            if current_lambda > mean_lambda + 3 * std_lambda:
                alert_triggered = True

        # Store the current rate for future moving statistics
        self.lambda_history.append(current_lambda)

        return alert_triggered
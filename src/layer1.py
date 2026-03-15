import pandas as pd
import numpy as np

class Layer1DataInjector:
    def __init__(self, csv_path, max_rate_lambda_star=0.05, baseline_rate=0.001):
        """
        Initializes the Data Streamer and the Non-Stationary Poisson Process (NSPP) attacker.
        
        :param csv_path: Path to the MetroPT3 CSV dataset.
        :param max_rate_lambda_star: The upper bound attack rate (lambda*) for the thinning algorithm.
        :param baseline_rate: The normal/minimum attack rate.
        """
        self.csv_path = csv_path
        # chunksize=1 allows us to stream the CSV row-by-row, simulating real-time acquisition
        self.data_iterator = pd.read_csv(csv_path, chunksize=1) 
        
        self.time_t = 0.0 # Simulated time (increments per sensor reading)
        
        # NSPP Parameters
        self.lambda_star = max_rate_lambda_star 
        self.baseline_rate = baseline_rate
        
        # Initialize the first potential arrival time for the thinning algorithm
        self.next_potential_attack_time = self._generate_next_arrival()

    def _generate_next_arrival(self):
        """
        Step 2 & 3 of the Thinning Algorithm:
        Generates the next potential arrival time using the maximum rate lambda_star.
        Math: t = t + [- (1/lambda*) * ln(U)]
        """
        u = np.random.uniform(0, 1)
        inter_arrival = - (1.0 / self.lambda_star) * np.log(u)
        return self.time_t + inter_arrival

    def lambda_t(self, current_time):
        """
        The dynamic intensity function lambda(t) representing the Non-Stationary attack rate.
        We use a sine wave to simulate "bursts" of attacks (e.g., higher attack frequency 
        during certain operational windows).
        """
        # Creates a smooth wave between baseline_rate and lambda_star
        rate = self.baseline_rate + (self.lambda_star - self.baseline_rate) * np.abs(np.sin(current_time / 1000.0))
        return rate

    def thinning_algorithm(self):
        """
        Steps 4 & 5 of the Thinning Algorithm:
        Decides whether to accept or reject the generated potential attack point based on 
        the time-varying probability lambda(t) / lambda*.
        """
        attack_injected = False
        
        # Process arrivals if the simulated time has reached the next potential attack time
        while self.time_t >= self.next_potential_attack_time:
            
            # Generate second U to accept or reject the point
            u2 = np.random.uniform(0, 1)
            current_rate = self.lambda_t(self.next_potential_attack_time)
            
            # Accept the attack with probability p = lambda(t) / lambda*
            if u2 <= (current_rate / self.lambda_star):
                attack_injected = True 
            
            # Generate the next potential arrival time for the loop
            self.next_potential_attack_time = self._generate_next_arrival()
            
        return attack_injected

    def inject_fdi(self, data_row):
        """
        False Data Injection (FDI): Maliciously alters the Pressure sensor (TP2).
        """
        attacked_row = data_row.copy()
        
        # Simulate a cyber attack causing a massive, unnatural spike in Pressure
        # Note: You may need to adjust the exact column name 'TP2' depending on your CSV headers
        spike_factor = np.random.uniform(2.5, 5.0) 
        
        if 'TP2' in attacked_row.columns:
            attacked_row['TP2'] = attacked_row['TP2'] * spike_factor
            
        return attacked_row

    def stream_data(self):
        """
        Yields the next row of data, orchestrating the stream and the attack injection.
        To be called continuously in a WHILE loop in main.py.
        """
        try:
            # Read the next sensor acquisition
            row = next(self.data_iterator)
            self.time_t += 1.0 # Advance simulated time by 1 unit per reading
            
            # Check if the mathematical model triggers an attack at this exact time
            is_attack = self.thinning_algorithm()
            
            if is_attack:
                row = self.inject_fdi(row)
                row['Ground_Truth_Label'] = "Cyber_Attack"
            else:
                row['Ground_Truth_Label'] = "Normal"
                
            return row
            
        except StopIteration:
            # Reached the end of the CSV file
            return None 
# Real-Time Cyber-Physical Intrusion Detection System (CP-IDS)

### Overview
This project implements a highly efficient, 3-layer **Cyber-Physical Intrusion Detection System (CP-IDS)** designed to simulate, detect, and physically verify sophisticated cyber-attacks on industrial machinery. 

Traditional security filters often assume that attacks arrive at a constant, stationary rate, which fails to capture the dynamic, "bursty" nature of real-world cyber threats. To address this, this project utilizes **Non-Stationary Poisson Processes** and **Maximum Likelihood Estimation (MLE)** to accurately track the accelerating frequency of attacks. Built and tested using the **MetroPT-3 (Air Compressor)** dataset, the algorithm successfully distinguishes between naturally occurring mechanical faults and malicious **False Data Injection (FDI)** attacks in real-time.

## Project Structure

ISS_Project/
│
├── src/
│   ├── main.py
│   ├── layer1.py
│   ├── layer2.py
│   └── layer3.py
│
├── data/              # dataset folder (not included in repo)
├── requirements.txt
├── README.md
└── .gitignore

### ⚙️ Architecture & Core Layers

**Layer 1: Mathematical Attack Injection (The Thinning Algorithm)**
Instead of injecting a constant stream of anomalies, this layer simulates a sophisticated, time-varying cyber-attack. It streams the raw CSV sensor data row-by-row to simulate a live acquisition feed. To model the unpredictable bursts of a real attacker, it uses the **Thinning Algorithm** to generate a Non-Stationary Poisson Process. By establishing an upper bound maximum attack rate ($\lambda^*$), the algorithm independently generates random variables to mathematically accept or reject potential attacks based on a dynamic intensity function $\lambda(t)$. When an attack is accepted, an FDI spike is injected into the Pressure sensor (`TP2`).

**Layer 2: Statistical Anomaly Detection (MLE & 3-Sigma Rule)**
This layer acts as the mathematical filter. Because cyber-attacks are rare but severe events, this layer continuously monitors the frequency of anomalies using a moving time window. When raw pressure spikes are detected, it uses **Maximum Likelihood Estimation (MLE)** to fit the non-stationary intensity function $\lambda(t) = a + bt$ to the arrival times of the anomalies. If the estimated attack frequency unexpectedly accelerates and breaks a 3-standard-deviation threshold, Layer 2 triggers a statistical cyber-attack alert.

**Layer 3: Physics-Based Consistency Engine**
To eliminate the high rate of false positives common in statistical anomaly detection, Layer 3 applies the physical laws of the machine. When Layer 2 flags a pressure anomaly, Layer 3 cross-checks the current physical baselines of the `Motor_current` and `Oil_temperature`. If the pressure is spiking but the motor is perfectly fine, the system recognizes a violation of physical laws—proving the alert is a **Sensor Spoofing / Cyber Attack (FDI)** rather than a mechanical hardware fault.

### 🚀 Performance & Results
*   **Real-Time Processing:** The system processes data, solves the MLE optimizations, and evaluates physical thresholds with an average latency of **< 5 milliseconds per sensor reading**, mathematically proving its viability for live, real-time industrial deployment.
*   **High Accuracy Detection:** In a 30,000-row simulation, the system flagged anomalies at a rate of **~5.7%**, perfectly mirroring the theoretical upper bound attack rate parameter ($\lambda^* = 0.05$) set in the Thinning Algorithm.
*   **Visual Analytics:** The project includes an automated `matplotlib` graphing orchestrator that visually maps the raw injected attacks (Layer 1), the dynamic MLE estimations (Layer 2), and the final confirmed System Alerts against the Ground Truth (Layer 3).

### 🛠️ Tech Stack
*   **Python 3.x**
*   **Pandas:** For chunk-based, real-time data streaming (`chunksize=1`).
*   **SciPy (`scipy.optimize`):** For executing the Maximum Likelihood Estimation (MLE) using SLSQP bounds and constraints.
*   **NumPy:** For mathematical Poisson generation and rolling 3-sigma statistical baselines.
*   **Matplotlib:** For multi-axis visualization of the non-stationary intensity functions.


## Installation

Clone the repository:

git clone https://github.com/zainabzahara/ISS_Project.git

Move into the project:

cd ISS_Project

Install dependencies:

pip install -r requirements.txt


## Dataset

The dataset is not included due to GitHub file size limits.

Download the **MetroPT Air Compressor dataset** and place it inside:

data/


## Running the Project

Run the main script:

python src/main.py

## Author

Zainab Zahara  
Master’s in Electronics Engineering  
University of Bologna
# Real-Time Cyber-Physical Intrusion Detection System (CP-IDS)

### Overview
This project implements a highly efficient, 3-layer **Cyber-Physical Intrusion Detection System (CP-IDS)** designed to simulate, detect, and physically verify sophisticated cyber-attacks on industrial machinery. 

Traditional security filters often assume that attacks arrive at a constant, stationary rate, which fails to capture the dynamic, "bursty" nature of real-world cyber threats. To address this, this project utilizes **Non-Stationary Poisson Processes** and **Maximum Likelihood Estimation (MLE)** to accurately track the accelerating frequency of attacks. Built and tested using the **MetroPT-3 (Air Compressor)** dataset, the algorithm successfully distinguishes between naturally occurring mechanical faults and malicious **False Data Injection (FDI)** attacks in real-time.


## Project Structure

```
ISS_Project/
│
├── src/
│   ├── main.py
│   ├── layer1.py
│   ├── layer2.py
│   └── layer3.py
│
├── data/                # dataset folder (not included in repository)
│
├── Results/
│   ├──results_T1000_A2
│   ├──results_T500_A2
│   ├──results_T100_A2
├── requirements.txt
├── README.md
└── .gitignore
```

### ⚙️ Architecture & Core Layers

**Layer 1: Mathematical Attack Injection (The Thinning Algorithm)**
Handles dataset and mathematical attack generation using Thinning algorithm for implementaion of Non-Stationary Poisson Process.

**Layer 2: Statistical Anomaly Detection (MLE & 3-Sigma Rule)**
This layer acts as the mathematical filter. Because cyber-attacks are rare but severe events, this layer continuously monitors the frequency of anomalies using a moving time window. 

**Layer 3: Physics-Based Consistency Engine**
This layer contains the physical laws of the machine. It only runs, When Layer 2 triggers an alert.

### Performance & Results
The system successfully integrates three complementary mechanisms:

* **Layer 1:** Generates realistic non-stationary cyber-attacks

* **Layer 2:** Detects statistical anomalies using real-time MLE estimation

* **Layer 3:** Validates anomalies using physical system constraints

Together, these layers enable high-accuracy real-time intrusion detection in cyber-physical systems.


Note: To maintain repository performance and adhere to storage limits, only the selective images and statistical CSV summaries are included.

### Tech Stack
*   **Python 3.x**
*   **Pandas:** For chunk-based, real-time data streaming (`chunksize=1`).
*   **SciPy (`scipy.optimize`):** For executing the Maximum Likelihood Estimation (MLE) using SLSQP bounds and constraints.
*   **NumPy:** For mathematical Poisson generation and rolling 3-sigma statistical baselines.
*   **Matplotlib:** For multi-axis visualization of the non-stationary intensity functions.


## Installation

Clone the repository: 
```
git clone https://github.com/zainabzahara/ISS_Project.git
```

Move into the project:
```
cd ISS_Project
```

Install dependencies:
```
pip install -r requirements.txt
```

## Dataset

The dataset is not included due to GitHub file size limits.

Download the **MetroPT Air Compressor dataset** and place it inside:

data/ 'https://archive.ics.uci.edu/dataset/791/metropt+3+dataset'

## Running the Project

Run the main script:
```
python src/main.py
```

## Author

Zainab Zahara  
Master’s in Electronics Engineering  
University of Bologna

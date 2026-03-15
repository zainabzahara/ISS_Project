import time
import pandas as pd
import matplotlib.pyplot as plt


#Importing the 3 laeyers
from layer1 import Layer1DataInjector
from layer2 import Layer2StatDetector
from layer3 import Layer3PhysicsValidator


def main():
    # 1. Configuration & Paths
    # Since main.py is in 'src', we go up one level to 'data' to find the CSV
    CSV_PATH = "../data/MetroPT3(AirCompressor).csv"  
    
    # The Real-Time constraint: 
    # Processing must be faster than 50 milliseconds per sensor reading
    MAX_PROCESSING_TIME = 0.05  
    
    # Define how long to run the simulation before generating graphs
    MAX_ITERATIONS = 30000 

    print("Initializing Cyber-Physical Intrusion Detection System...")
    
    # 2. Initialize the Architecture
    # Layer 1: Data stream and mathematical attack generator (Thinning Algorithm)
    layer1 = Layer1DataInjector(csv_path=CSV_PATH, max_rate_lambda_star=0.05, baseline_rate=0.001)
    
    # Layer 2: Maximum Likelihood Estimation (MLE) filter
    layer2 = Layer2StatDetector(time_window=1000, lambda_history_size=50)
    
    # Layer 3: Physical consistency engine (Motor vs Pressure)
    layer3 = Layer3PhysicsValidator(history_window=500)
    
    print("System Online. Starting Real-Time Stream...\n")
    print("-" * 65)
    
    # 3. Tracking Variables
    t = 0
    anomalies_detected = 0
    total_processing_time = 0.0
    
    # --- GRAPHING DATA STORAGE ---
    plot_time = []
    plot_tp2 = []
    plot_lambda = []
    plot_ground_truth = []
    plot_l3_alerts = []


    try:
        while t < MAX_ITERATIONS:
            # -------------------------------------------------------------
            # LAYER 1: Fetch next data acquisition (and potential attack)
            # -------------------------------------------------------------
            data_row_df = layer1.stream_data()
            if data_row_df is None:
                print("End of data stream reached.")
                break
                
            # Convert the 1-row dataframe chunk to a pandas Series for easy access
            data_row = data_row_df.squeeze()
            
            # --- START REAL-TIME TIMER ---
            # time.perf_counter() is highly accurate for benchmarking code speed
            start_time = time.perf_counter() 
            
            # -------------------------------------------------------------
            # LAYER 2: Statistical Filter (MLE & 3-Sigma Rule)
            # -------------------------------------------------------------
            is_anomaly = layer2.check_statistical_alert(data_row, time_t=t)
            
            # -------------------------------------------------------------
            # LAYER 3: Physics-Based Consistency Engine
            # -------------------------------------------------------------
            if is_anomaly:
                anomalies_detected += 1
                
                # Ask the Physics Engine to diagnose the root cause
                diagnosis = layer3.classify_root_cause(data_row)
                
                print(f"[Time step: {t}] L2 STATISTICAL ALERT on TP2 Pressure!")
                print(f"   {diagnosis}")
                print(f"   Ground Truth Tag: {data_row['Ground_Truth_Label']}\n")

                plot_l3_alerts.append(t) # Save the time the alert happened

            else:
                # If the system is operating normally, update the physical 
                # baseline memory so the AI knows what "normal" currently looks like
                layer3.update_baselines(data_row)
            
            # --- STOP TIMER ---
            end_time = time.perf_counter()
            
            # 4. Performance Measurement
            processing_time = end_time - start_time
            total_processing_time += processing_time
            
            # Validate the Real-Time Constraint
            if processing_time > MAX_PROCESSING_TIME:
                print(f"[WARNING] Latency Spike at t={t}: {processing_time:.4f} seconds!")
            # -------------------------------------------------------------
            # --- SAVE DATA FOR GRAPHING ---
            plot_time.append(t)
            plot_tp2.append(data_row['TP2'])
            # 1 if Attack, 0 if Normal
            plot_ground_truth.append(1 if data_row['Ground_Truth_Label'] == "Cyber_Attack" else 0) 
            
            # Get the current estimated mathematical attack rate from Layer 2's memory
            current_estimated_lambda = layer2.lambda_history[-1] if len(layer2.lambda_history) > 0 else 0
            plot_lambda.append(current_estimated_lambda)
            # -------------------------------------------------------------

            t += 1
            
            # Print a "Heartbeat" every 5000 iterations to show it is running smoothly
            if t % 5000 == 0:
                avg_time = (total_processing_time / t) * 1000 # Convert to milliseconds
                print(f"--- Heartbeat: Processed {t} rows. Avg Speed: {avg_time:.3f} ms/loop ---")

    except KeyboardInterrupt:
        # Allows you to stop the script cleanly by pressing Ctrl+C in the terminal
        print("\nStream manually interrupted by user.")
        
    # 5. Final Report Generation
    print("-" * 65)
    print("Simulation Complete.")
    print(f"Total Acquisitions Processed : {t}")
    print(f"Total Anomalies Flagged      : {anomalies_detected}")
    
    if t > 0:
        final_avg_speed = (total_processing_time / t) * 1000
        print(f"Average Processing Latency   : {final_avg_speed:.3f} ms per acquisition")
        print("\nCONCLUSION: Real-Time Constraint Met Successfully." if final_avg_speed < (MAX_PROCESSING_TIME * 1000) else "Real-Time Constraint Failed.")


        # ---------------------------------------------------------
    # GENERATE VISUALIZATIONS FOR THE PROFESSOR
    # ---------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Graph 1: Layer 1 - The Physical Data & Attacks
    ax1.plot(plot_time, plot_tp2, color='royalblue', label='Sensor Data (TP2)', linewidth=1)
    ax1.set_title("Layer 1: Sensor Data Stream & Injected Attacks", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Pressure Value")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Graph 2: Layer 2 - The Mathematical MLE Estimation
    ax2.plot(plot_time, plot_lambda, color='darkorange', label='Estimated Arrival Rate λ(t)', linewidth=1.5)
    ax2.set_title("Layer 2: Statistical Estimation of Attack Rate (Non-Stationary Poisson)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Estimated λ(t)")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Graph 3: Layer 3 - Alerts vs Ground Truth
    ax3.plot(plot_time, plot_ground_truth, color='red', label='Ground Truth (Actual Attack)', linestyle='--', linewidth=2)
    
    # Draw vertical lines everywhere Layer 3 sounded the alarm
    for alert_t in plot_l3_alerts:
        ax3.axvline(x=alert_t, color='black', alpha=0.4, linewidth=1, label='System Alert' if alert_t == plot_l3_alerts else "")
        
    ax3.set_title("Layer 3: System Alerts vs Actual Attacks (Ground Truth)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Attack (1=Yes, 0=No)")
    ax3.set_xlabel("Time Step (Sensor Reading)")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

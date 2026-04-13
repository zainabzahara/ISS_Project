import time
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    #MAX_ITERATIONS = 30000
    # changing her a bit 
    # We set a very high number now, as we will save in chunks
    MAX_ITERATIONS = 1000000 
    SAVE_INTERVAL = 20000 # Save a graph every 20k rows



    print("Initializing Cyber-Physical Intrusion Detection System...")
    
    # 2. Initialize the Architecture
    # Layer 1: Data stream and mathematical attack generator (Thinning Algorithm)
    layer1 = Layer1DataInjector(csv_path=CSV_PATH, max_rate_lambda_star=0.05, baseline_rate=0.001)
    
    # Layer 2: Maximum Likelihood Estimation (MLE) filter
    layer2 = Layer2StatDetector(time_window=100, lambda_history_size=50)
    
    # Layer 3: Physical consistency engine (Motor vs Pressure)
    layer3 = Layer3PhysicsValidator(history_window=500)
    
    print("System Online. Starting Real-Time Stream...\n")
    print("-" * 65)
    
    # 3. Tracking Variables
    t = 0
    anomalies_detected = 0
    total_processing_time = 0.0
    current_chunk_cyber = 0
    current_chunk_hardware = 0
    
    # --- GRAPHING DATA STORAGE ---
    plot_time = []
    plot_tp2 = []
    plot_lambda = []
    plot_ground_truth = []
    plot_l3_alerts = []

    # --- ADD THIS HERE ---
    # This finds the "parent" folder (ISS_project) and makes a results folder there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_path = os.path.join(project_root, "results_T100_A2")
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # --- Statistical Trackers ---
    TP = 0  # True Positives (System caught an actual attack)
    FP = 0  # False Positives (System cried wolf on normal/hardware data)
    FN = 0  # False Negatives (System missed a real attack)
    TN = 0  # True Negatives (System correctly stayed quiet)
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
            
            system_alert = False # Default state
            actual_is_attack = (data_row['Ground_Truth_Label'] == "Cyber_Attack")

            # -------------------------------------------------------------
            # LAYER 3: Physics-Based Consistency Engine
            # -------------------------------------------------------------
            if is_anomaly:
                anomalies_detected += 1
                
                # Ask the Physics Engine to diagnose the root cause
                diagnosis = layer3.classify_root_cause(data_row)
                
                #print(f"[Time step: {t}] L2 STATISTICAL ALERT on TP2 Pressure!")
                #print(f"   {diagnosis}")
                #print(f"   Ground Truth Tag: {data_row['Ground_Truth_Label']}\n")
                
                # We count them internally instead of printing every line
                if "CYBER ATTACK" in diagnosis:
                    system_alert = True
                    current_chunk_cyber += 1
                    plot_l3_alerts.append(t) # Save the time the alert happened
                else:
                    system_alert = False # It was a Hardware Fault
                    current_chunk_hardware += 1
                #plot_l3_alerts.append(t) # Save the time the alert happened

            else:
                # If the system is operating normally, update the physical 
                # baseline memory so the AI knows what "normal" currently looks like
                layer3.update_baselines(data_row)
                system_alert = False
            # --- STOP TIMER ---
            end_time = time.perf_counter()
            
            # --- Calculate TP, FP, FN, TN ---
            if system_alert and actual_is_attack:
                TP += 1
            elif system_alert and not actual_is_attack:
                FP += 1
            elif not system_alert and actual_is_attack:
                FN += 1
            else:
                TN += 1

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
            plot_lambda.append(layer2.lambda_history[-1] if layer2.lambda_history else 0)

            # Get the current estimated mathematical attack rate from Layer 2's memory
            #current_estimated_lambda = layer2.lambda_history[-1] if len(layer2.lambda_history) > 0 else 0
            #plot_lambda.append(current_estimated_lambda)
            # -------------------------------------------------------------

            # -------------------------------------------------------------
            # NEW: CHUNK SAVING LOGIC
            # -------------------------------------------------------------
            if (t + 1) % SAVE_INTERVAL == 0:
                print(f"\n--- [REPORT FOR CHUNK {t+1-SAVE_INTERVAL} TO {t+1}] ---")
                print(f"  > Cyber Attacks Detected: {current_chunk_cyber}")
                print(f"  > Hardware Faults Found : {current_chunk_hardware}")
                print(f"  > Saving Plot to results/simulation_at_{t+1}.png...")
                
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                ax1.plot(plot_time, plot_tp2, color='royalblue', linewidth=1)
                ax1.set_title(f"Sensor Stream (Chunk: {t-SAVE_INTERVAL} to {t})", fontweight='bold')
                ax1.set_ylabel("Pressure")

                ax2.plot(plot_time, plot_lambda, color='darkorange', linewidth=1.5)
                ax2.set_title("Estimated Attack Rate λ(t)", fontweight='bold')
                #old line
                #ax3.plot(plot_time, plot_ground_truth, color='red', linestyle='--', label='Actual')

                # NEW line (Makes the red line FAT and BRIGHT):
                ax3.plot(plot_time, plot_ground_truth, color='red', linewidth=4, alpha=0.3, label='Actual')
                # Only plot alerts relevant to this current chunk
                current_alerts = [a for a in plot_l3_alerts if (t - SAVE_INTERVAL) <= a <= t]
                for alert_t in current_alerts:
                    ax3.axvline(x=alert_t, color='black', alpha=0.3)
                
                ax3.set_title("Alerts vs Ground Truth", fontweight='bold')
                ax3.set_xlabel("Time Step")

                plt.tight_layout()
                plt.savefig(os.path.join(results_path, f"simulation_at_{t+1}.png"))
                
                # IMPORTANT: Clear memory
                plt.close(fig) # Closes the figure window
                plt.clf()      # Clears the current figure
                
                # Reset counters and data storage for the next chunk
                current_chunk_cyber = 0
                current_chunk_hardware = 0

                # Empty arrays to keep latency low
                plot_time, plot_tp2, plot_lambda, plot_ground_truth, plot_l3_alerts = [], [], [], [], []
                print(f"--- Memory Cleared. Latency Restored. ---\n")

            t += 1
            if t % 5000 == 0:
                avg_time = (total_processing_time / t) * 1000
                print(f"Heartbeat: {t} rows. Speed: {avg_time:.3f} ms/loop")

    except KeyboardInterrupt:
        print("\nStream manually interrupted.")
    # --- Calculate Final Metrics ---
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n=== FINAL PERFORMANCE FOR T={layer2.time_window} ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")

    # Export to CSV
    csv_path = os.path.join(results_path, f"summary_stats_T{layer2.time_window}.csv")
    with open(csv_path, "w") as f:
        f.write("Metric,Value\n")
        f.write(f"Total Rows Processed,{t}\n")
        f.write(f"True Positives (TP),{TP}\n")
        f.write(f"False Positives (FP),{FP}\n")
        f.write(f"False Negatives (FN),{FN}\n")
        f.write(f"True Negatives (TN),{TN}\n")
        f.write(f"Precision,{precision:.4f}\n")
        f.write(f"Recall,{recall:.4f}\n")
        f.write(f"F1-Score,{f1_score:.4f}\n")
    print(f"Simulation Complete. Total Rows: {t}")

if __name__ == "__main__":
    main()
    
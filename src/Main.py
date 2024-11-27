import tkinter as tk
from tkinter import messagebox
import subprocess
import os

def run_script(script_name):
    """Run the specified Python script."""
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        subprocess.run(["python", script_path], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {script_name}: {e}")

def open_clustering_menu():
    """Open the Clustering menu."""
    clustering_window = tk.Toplevel(root)
    clustering_window.title("Clustering Menu")

    tk.Label(clustering_window, text="Choose a Clustering Option").pack(pady=10)

    tk.Button(clustering_window, text="1. Agglomerative",
              command=lambda: run_script("BossJore\PycharmProjects\Airline_Passenger_Satisfaction\Clustering\Agglomerative.py")).pack(pady=5)
    tk.Button(clustering_window, text="2. KMEANS",
              command=lambda: run_script("../Clustering/KMEANS.py")).pack(pady=5)
    tk.Button(clustering_window, text="3. Comparison",
              command=lambda: run_script("../Clustering/Comparison.py")).pack(pady=5)
    tk.Button(clustering_window, text="Back", command=clustering_window.destroy).pack(pady=10)

def open_data_menu():
    """Open the Data menu."""
    data_window = tk.Toplevel(root)
    data_window.title("Data Menu")

    tk.Label(data_window, text="Choose a Data Option").pack(pady=10)

    tk.Button(data_window, text="1. Clean Data",
              command=lambda: run_script("../data/Data_cleaning.py")).pack(pady=5)
    tk.Button(data_window, text="2. Normalize Data",
              command=lambda: run_script("../data/Normalized_data.py")).pack(pady=5)
    tk.Button(data_window, text="3. Reduce Data",
              command=lambda: run_script("../data/Normalized_data2.py")).pack(pady=5)
    tk.Button(data_window, text="Back", command=data_window.destroy).pack(pady=10)

def exit_program():
    """Exit the program."""
    root.destroy()

# Main GUI
root = tk.Tk()
root.title("Airline Passenger Satisfaction")

tk.Label(root, text="Main Menu", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root, text="1. Clustering", width=20, command=open_clustering_menu).pack(pady=10)
tk.Button(root, text="2. Data", width=20, command=open_data_menu).pack(pady=10)
tk.Button(root, text="3. Exit", width=20, command=exit_program).pack(pady=10)

# Run the main loop
root.mainloop()

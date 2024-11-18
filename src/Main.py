import tkinter as tk
from tkinter import messagebox
import subprocess
import sklearn



def run_data_cleaning():
    try:
        subprocess.run(['python', 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Data_cleaning.py'], check=True)
        messagebox.showinfo("Success", "Data cleaned and saved successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while cleaning data: {e}")


def run_normalization():
    try:
        subprocess.run(['python', 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_data.py'], check=True)
        messagebox.showinfo("Success", "Data normalized successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while normalizing data: {e}")


def run_data_reduction():
    try:
        subprocess.run(['python', 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Normalized_data2.py'], check=True)
        messagebox.showinfo("Success", "Data reduced to 5000 rows and saved successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while reducing data: {e}")


def run_kmeans():
    try:
        subprocess.run(['python', 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/Clustering/KMEANS.py'], check=True)
        messagebox.showinfo("Success", "KMeans clustering executed successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running KMeans clustering: {e}")


def run_agglomerative():
    try:
        subprocess.run(['python', 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/Clustering/Agglomerative.py'], check=True)
        messagebox.showinfo("Success", "Agglomerative clustering executed successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running Agglomerative clustering: {e}")


def run_comparison():
    try:
        subprocess.run(['python', 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/Clustering/Comparison.py'], check=True)
        messagebox.showinfo("Success", "Clustering comparison executed successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while comparing clustering methods: {e}")


def main_menu():
    root = tk.Tk()
    root.title("Airline Passenger Satisfaction Analysis")

    # Data Menu
    data_menu = tk.LabelFrame(root, text="Data Operations", padx=10, pady=10)
    data_menu.grid(row=0, column=0, padx=10, pady=10)

    tk.Button(data_menu, text="Clean Data", width=20, command=run_data_cleaning).grid(row=0, column=0, padx=5, pady=5)
    tk.Button(data_menu, text="Normalize Data", width=20, command=run_normalization).grid(row=1, column=0, padx=5, pady=5)
    tk.Button(data_menu, text="Reduce Data", width=20, command=run_data_reduction).grid(row=2, column=0, padx=5, pady=5)

    # Clustering Menu
    clustering_menu = tk.LabelFrame(root, text="Clustering Operations", padx=10, pady=10)
    clustering_menu.grid(row=1, column=0, padx=10, pady=10)

    tk.Button(clustering_menu, text="KMeans Clustering", width=20, command=run_kmeans).grid(row=0, column=0, padx=5, pady=5)
    tk.Button(clustering_menu, text="Agglomerative Clustering", width=20, command=run_agglomerative).grid(row=1, column=0, padx=5, pady=5)
    tk.Button(clustering_menu, text="Compare Clustering", width=20, command=run_comparison).grid(row=2, column=0, padx=5, pady=5)

    # Exit Button
    tk.Button(root, text="Exit", width=20, command=root.quit).grid(row=2, column=0, padx=5, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main_menu()

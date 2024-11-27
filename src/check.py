import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for rendering plots to files
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageTk


def sanitize_filename(filename):
    """Sanitize column names to make them safe for use as filenames."""
    return filename.replace('/', '_').replace('\\', '_').replace(' ', '_')


def plot_column_distribution(df, column_name, output_dir):
    """Plot the distribution of values in a specified column and save the plot."""
    plt.figure(figsize=(8, 6))

    if df[column_name].dtype == 'object':  # Categorical column
        value_counts = df[column_name].value_counts()
        value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    else:  # Numerical column
        df[column_name].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')

    # Sanitize the column name to make it a valid file name
    safe_column_name = sanitize_filename(column_name)

    # Save the plot to the output directory
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_column_name}_distribution.png"))
    plt.close()  # Close the plot to avoid memory issues with large number of plots


def generate_plots(df, output_dir):
    """Generate plots for all columns and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    for column in df.columns:
        plot_column_distribution(df, column, output_dir)
        plot_files.append(f"{sanitize_filename(column)}_distribution.png")
    return plot_files


def open_plot(selected_plot, output_dir):
    """Open the selected plot file in a tkinter window."""
    try:
        image_path = os.path.join(output_dir, selected_plot)
        img = Image.open(image_path)
        img.thumbnail((600, 400))  # Resize for display
        img = ImageTk.PhotoImage(img)

        # Create a new top-level window to display the image
        plot_window = tk.Toplevel()
        plot_window.title(selected_plot)
        panel = tk.Label(plot_window, image=img)
        panel.image = img  # Keep a reference to avoid garbage collection
        panel.pack()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")


def choose_plot_window(plot_files, output_dir):
    """Display a window with a list of generated plots to choose from."""

    def on_select(event):
        selected_plot = plot_listbox.get(tk.ACTIVE)
        open_plot(selected_plot, output_dir)

    # Create the main Tkinter window
    window = tk.Tk()
    window.title("Select a Plot to View")

    # Listbox to show available plot files
    plot_listbox = tk.Listbox(window, height=15, width=50, selectmode=tk.SINGLE)
    plot_listbox.insert(tk.END, *plot_files)  # Add all plot filenames to the listbox
    plot_listbox.bind('<<ListboxSelect>>', on_select)

    # Add a scrollbar
    scrollbar = tk.Scrollbar(window, orient="vertical", command=plot_listbox.yview)
    plot_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    plot_listbox.pack(padx=10, pady=10)

    # Start the Tkinter main loop
    window.mainloop()


def main():
    # Load dataset
    file_path = 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Cleaned_Data.csv'
    df = pd.read_csv(file_path)

    # Create output directory for saving the plots
    output_dir = 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/plots'

    # Generate and save plots
    plot_files = generate_plots(df, output_dir)

    # Launch the Tkinter window to select and view the plots
    choose_plot_window(plot_files, output_dir)


if __name__ == "__main__":
    main()

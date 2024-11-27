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
    """Plot the distribution of values in a specified column as a histogram and save the plot."""
    plt.figure(figsize=(8, 6))

    if df[column_name].dtype == 'object':  # Categorical column (e.g., Gender, Customer Type)
        value_counts = df[column_name].value_counts()
        value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    else:  # Numerical column (e.g., Flight Distance, Age)
        df[column_name].plot(kind='hist', bins=20, color='skyblue', edgecolor='black', alpha=0.7)
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
    """Generate histograms for numerical columns and bar charts for categorical columns and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    for column in df.columns:
        plot_column_distribution(df, column, output_dir)
        plot_files.append(f"{sanitize_filename(column)}_distribution.png")
    return plot_files


def display_plots_in_grid(plot_files, output_dir, columns_per_row=3):
    """Display plots in a grid, three per row."""
    window = tk.Tk()
    window.title("All Plots in Grid")

    # Create a frame to hold the images in a grid
    frame = tk.Frame(window)
    frame.pack(padx=10, pady=10)

    # Create a variable to track the number of columns in the current row
    row = 0
    col = 0

    # Loop through the plot files and display them in a grid
    for plot_file in plot_files:
        image_path = os.path.join(output_dir, plot_file)
        img = Image.open(image_path)
        img.thumbnail((400, 300))  # Resize for display (adjust size if needed)
        img = ImageTk.PhotoImage(img)

        # Add the image to the current position in the grid
        label = tk.Label(frame, image=img)
        label.image = img  # Keep a reference to avoid garbage collection
        label.grid(row=row, column=col, padx=32, pady=16)

        # Move to the next column, and if we exceed the number of columns per row, move to the next row
        col += 1
        if col == columns_per_row:
            col = 0
            row += 1

    window.mainloop()


def main():
    # Load dataset
    file_path = 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/data/Cleaned_Data.csv'
    df = pd.read_csv(file_path)

    # Create output directory for saving the plots
    output_dir = 'C:/Users/BossJore/PycharmProjects/Airline_Passenger_Satisfaction/plots'

    # Generate and save plots
    plot_files = generate_plots(df, output_dir)

    # Display all the plots in a grid layout (three per row)
    display_plots_in_grid(plot_files, output_dir)


if __name__ == "__main__":
    main()

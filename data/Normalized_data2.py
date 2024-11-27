import pandas as pd

# Load the already normalized data
file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Normalized_Data.csv'
data = pd.read_csv(file_path)

# Randomly sample 5000 rows from the data
reduced_data = data.sample(n=5000)

# Save the reduced dataset to a new file
reduced_file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Normalized_data2.csv'
reduced_data.to_csv(reduced_file_path, index=False)

print("Data reduced to 5000 rows and saved to 'Normalized_data2.csv'.")

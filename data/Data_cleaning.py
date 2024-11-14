import pandas as pd

# File paths
input_file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Main_Data.csv'
output_file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Cleaned_Data.csv'

# Load the data
data = pd.read_csv(input_file_path)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Unnamed: 0', 'id'])

# Fill missing values in 'Arrival Delay in Minutes' with the median
median_value = data_cleaned['Arrival Delay in Minutes'].median()
data_cleaned['Arrival Delay in Minutes'] = data_cleaned['Arrival Delay in Minutes'].fillna(median_value)

# Save the cleaned data
data_cleaned.to_csv(output_file_path, index=False)

print(f"Data cleaned and saved to {output_file_path}")

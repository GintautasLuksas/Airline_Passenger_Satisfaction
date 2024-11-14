import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Cleaned_Data.csv'
data = pd.read_csv(file_path)

# Initialize MinMaxScaler for normalization
scaler = MinMaxScaler()

# Normalize Age (7 to 85 range)
data['Age'] = scaler.fit_transform(data[['Age']])

# Normalize Flight Distance (31 to 4983 range)
data['Flight Distance'] = scaler.fit_transform(data[['Flight Distance']])

# Normalize the 14 satisfaction-related columns (assuming they are columns 6 to 19 in the dataset)
satisfaction_columns = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
    'Inflight service', 'Cleanliness'
]
data[satisfaction_columns] = scaler.fit_transform(data[satisfaction_columns])

# Create the satisfaction score (average of 14 satisfaction-related columns)
data['Satisfaction Score'] = data[satisfaction_columns].mean(axis=1)

# Correctly sum the Departure and Arrival Delay to get Total Delay Time (No scaling here)
data['Total Delay Time'] = data['Departure Delay in Minutes'] + data['Arrival Delay in Minutes']

# Normalize Gender: Male=0, Female=1
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Normalize Customer Type: Loyal=1, Disloyal=0
data['Customer Type'] = data['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)

# Normalize Type of Travel: Business=1, Personal=0
data['Type of Travel'] = data['Type of Travel'].apply(lambda x: 1 if x == 'Business travel' else 0)

# Normalize Class: Eco=1, Eco Plus=2, Business=0
data['Class'] = data['Class'].apply(lambda x: 2 if x == 'Eco Plus' else (1 if x == 'Eco' else 0))

# Normalize Satisfaction: Satisfied=1, Neutral or Dissatisfied=0
data['satisfaction'] = data['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Round all numerical columns to 3 decimal places
numerical_columns = [
    'Age', 'Flight Distance', 'Satisfaction Score', 'Total Delay Time'
]  # Only include the columns that need rounding

data[numerical_columns] = data[numerical_columns].round(3)

# Drop the original 14 satisfaction-related columns
data.drop(columns=satisfaction_columns, inplace=True)

# Save the normalized data to a new CSV file
normalized_file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Normalized_Data.csv'
data.to_csv(normalized_file_path, index=False)

print("Normalization complete. Data saved to 'Normalized_Data.csv'.")

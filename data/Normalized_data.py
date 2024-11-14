import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Cleaned_Data.csv'
data = pd.read_csv(file_path)

# Initialize MinMaxScaler for normalization
scaler = MinMaxScaler()

# 1. Map Gender: Male (0), Female (1)
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# 2. Map Customer Type: Loyal (1), Disloyal (0)
data['Customer Type'] = data['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)

# 3. Categorize Age into bins: 0-20 (0), 21-40 (1), 41-60 (2), 61-85 (3)
age_bins = [0, 20, 40, 60, 85]
age_labels = [0, 1, 2, 3]
data['Age'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=True)

# 4. Map Type of Travel: Business (1), Personal (0)
data['Type of Travel'] = data['Type of Travel'].apply(lambda x: 1 if x == 'Business travel' else 0)

# 5. Map Class: Eco Plus (2), Eco (1), Business (0)
data['Class'] = data['Class'].apply(lambda x: 2 if x == 'Eco Plus' else (1 if x == 'Eco' else 0))

# 6. Normalize Flight Distance
data['Flight Distance'] = scaler.fit_transform(data[['Flight Distance']])

# 7. Calculate average satisfaction score from the 14 satisfaction-related columns
satisfaction_columns = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
    'Inflight service', 'Cleanliness'
]
data['Satisfaction Score'] = data[satisfaction_columns].mean(axis=1)

# 8. Normalize Departure Delay
data['Departure Delay in Minutes'] = scaler.fit_transform(data[['Departure Delay in Minutes']])

# 9. Normalize Arrival Delay
data['Arrival Delay in Minutes'] = scaler.fit_transform(data[['Arrival Delay in Minutes']])

# 10. Map Satisfaction Outcome: satisfied (1), neutral or dissatisfied (0)
data['satisfaction'] = data['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# Save the transformed data to a new CSV file
normalized_file_path = r'C:\Users\BossJore\PycharmProjects\Airline_Passenger_Satisfaction\data\Normalized_Data.csv'
data.to_csv(normalized_file_path, index=False)

print("Data transformation complete. Transformed data saved to 'Normalized_Data.csv'.")

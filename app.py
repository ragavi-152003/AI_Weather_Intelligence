
import requests
import pandas as pd
import os

# Coimbatore location
latitude = 11.0168
longitude = 76.9558

# Historical weather API
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2026-09-27",
    "end_date": "2026-10-26",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m"
    ],
    "timezone": "Asia/Kolkata"
}

# Get weather data
response = requests.get(url, params=params)
data = response.json()

# Create DataFrame
df = pd.DataFrame({
    "Date_Time": data["hourly"]["time"],
    "Temperature": data["hourly"]["temperature_2m"],
    "Humidity": data["hourly"]["relative_humidity_2m"],
    "Rainfall": data["hourly"]["precipitation"],
    "Wind_Speed": data["hourly"]["wind_speed_10m"]
})

# Create folder
os.makedirs("data", exist_ok=True)

# Save to Excel
df.to_excel("data/weather_data.xlsx", index=False)

print(df.head())
print("Weather data saved to Excel successfully!")


# Read Excel file
df = pd.read_excel("data/weather_data.xlsx")

# Display dataset information
print("Total Records:", len(df))

# Calculate average temperature
print("Average Temperature:", df["Temperature"].mean())

# Find maximum temperature
print("Maximum Temperature:", df["Temperature"].max())

# Find minimum temperature
print("Minimum Temperature:", df["Temperature"].min())

# Calculate average humidity
print("Average Humidity:", df["Humidity"].mean())

# Calculate total rainfall
print("Total Rainfall:", df["Rainfall"].sum())


import matplotlib.pyplot as plt

df["Date_Time"] = pd.to_datetime(df["Date_Time"])

plt.figure(figsize=(10, 5))

plt.plot(df["Date_Time"], df["Temperature"])

plt.title("Coimbatore Temperature Analysis")
plt.xlabel("Date and Time")
plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/temperature_graph.png")

plt.show()


# Rainfall visualization
plt.figure(figsize=(10, 5))

plt.plot(df["Date_Time"], df["Rainfall"])

plt.title("Coimbatore Rainfall Analysis")
plt.xlabel("Date and Time")
plt.ylabel("Rainfall (mm)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/rainfall_graph.png")

plt.show()


# Daily average temperature
df["Date"] = df["Date_Time"].dt.date

daily_temp = df.groupby("Date")["Temperature"].mean()

# Create bar chart
plt.figure(figsize=(12, 5))

plt.bar(daily_temp.index.astype(str), daily_temp.values)

plt.title("Daily Average Temperature - Coimbatore")
plt.xlabel("Date")
plt.ylabel("Average Temperature (°C)")

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("data/daily_temperature_graph.png")

plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Select input and output
X = df[["Humidity", "Wind_Speed"]]
y = df["Temperature"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict temperature
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Enter weather conditions
humidity = float(input("Enter humidity (%): "))
wind_speed = float(input("Enter wind speed (km/h): "))

# Predict temperature
new_data = pd.DataFrame({
    "Humidity": [humidity],
    "Wind_Speed": [wind_speed]
})

predicted_temp = model.predict(new_data)

print("Predicted Temperature:", round(predicted_temp[0], 2), "°C")


# Actual vs Predicted Temperature
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature")

plt.savefig("data/actual_vs_predicted.png")

plt.show()

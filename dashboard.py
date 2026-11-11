
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Coimbatore Weather Intelligence Dashboard")

# Read Excel data
df = pd.read_excel("data/weather_data.xlsx")
df["Date_Time"] = pd.to_datetime(df["Date_Time"])

# Display weather statistics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Temperature", f"{df['Temperature'].mean():.2f} °C")
col2.metric("Maximum Temperature", f"{df['Temperature'].max():.2f} °C")
col3.metric("Average Humidity", f"{df['Humidity'].mean():.2f} %")
col4.metric("Total Rainfall", f"{df['Rainfall'].sum():.2f} mm")

# Graph 1: Temperature
st.subheader("1. Temperature Analysis")

fig1 = px.line(
    df,
    x="Date_Time",
    y="Temperature",
    title="Temperature Changes"
)
st.plotly_chart(fig1, use_container_width=True)

# Graph 2: Rainfall
st.subheader("2. Rainfall Analysis")

fig2 = px.line(
    df,
    x="Date_Time",
    y="Rainfall",
    title="Hourly Rainfall"
)
st.plotly_chart(fig2, use_container_width=True)

# Graph 3: Daily Average Temperature
st.subheader("3. Daily Average Temperature")

df["Date"] = df["Date_Time"].dt.date
daily_temp = df.groupby("Date", as_index=False)["Temperature"].mean()

fig3 = px.bar(
    daily_temp,
    x="Date",
    y="Temperature",
    title="Daily Average Temperature"
)
st.plotly_chart(fig3, use_container_width=True)

# Graph 4: Actual vs Predicted Temperature
st.subheader("4. Actual vs Predicted Temperature")

X = df[["Humidity", "Wind_Speed"]]
y = df["Temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Temperature": y_test.values,
    "Predicted Temperature": y_pred
})

fig4 = px.scatter(
    results,
    x="Actual Temperature",
    y="Predicted Temperature",
    title="Actual vs Predicted Temperature"
)
st.plotly_chart(fig4, use_container_width=True)

# Display weather data
st.subheader("Weather Dataset")
st.dataframe(df)

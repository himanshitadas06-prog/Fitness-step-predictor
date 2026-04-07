import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Fitness App", page_icon="🏃", layout="centered")

# Load model
model = pickle.load(open("stepcount_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏃 Fitness Journey Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict your daily step count easily</h4>", unsafe_allow_html=True)

st.write("---")

# Input section
st.subheader("📋 Enter Your Activity Details")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("🚶 Distance (km)", min_value=0.0)
    flights_climbed = st.number_input("🏢 Flights Climbed", min_value=0)
    goal = st.number_input("🎯 Step Goal", min_value=0)

with col2:
    active_time = st.number_input("⏱ Active Time (minutes)", min_value=0)
    calories = st.number_input("🔥 Calories Burned", min_value=0)
    perc_goal_completed = st.number_input("📊 % Goal Completed", min_value=0.0)

st.write("---")

# Prediction button
if st.button("🔮 Predict Step Count"):

    data = np.array([[distance, active_time, flights_climbed,
                      calories, goal, perc_goal_completed]])

    data = scaler.transform(data)
    result = model.predict(data)

    st.success(f"🎉 Predicted Step Count: {int(result[0])}")

    # Progress bar
    progress = min(int(perc_goal_completed), 100)
    st.progress(progress)

    # Message
    if progress < 50:
        st.warning("⚠️ Keep moving! You're below half your goal.")
    elif progress < 100:
        st.info("👍 Good job! You're getting close to your goal.")
    else:
        st.success("🏆 Excellent! Goal achieved!")

st.write("---")

# Footer
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
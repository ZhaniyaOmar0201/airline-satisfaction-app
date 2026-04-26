import streamlit as st
import pandas as pd
import joblib
import shap

# --- LOAD MODEL ---
model = joblib.load("model.pkl")

st.title("✈️ Airline Passenger Satisfaction")

st.write("Enter passenger details:")

# --- INPUTS ---
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
age = st.slider("Age", 10, 80, 30)
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
flight_distance = st.number_input("Flight Distance", 0, 5000, 500)

wifi = st.slider("Inflight wifi service", 0, 5, 3)
time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
booking = st.slider("Ease of Online booking", 0, 5, 3)
gate = st.slider("Gate location", 0, 5, 3)
food = st.slider("Food and drink", 0, 5, 3)
boarding = st.slider("Online boarding", 0, 5, 3)
seat = st.slider("Seat comfort", 0, 5, 3)
entertainment = st.slider("Inflight entertainment", 0, 5, 3)
onboard = st.slider("On-board service", 0, 5, 3)
legroom = st.slider("Leg room service", 0, 5, 3)
baggage = st.slider("Baggage handling", 0, 5, 3)
checkin = st.slider("Checkin service", 0, 5, 3)
inflight = st.slider("Inflight service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)

dep_delay = st.number_input("Departure Delay (min)", 0, 1000, 0)
arr_delay = st.number_input("Arrival Delay (min)", 0, 1000, 0)

# --- DATAFRAME ---
input_data = pd.DataFrame([{
    "Gender": gender,
    "Customer Type": customer_type,
    "Age": age,
    "Type of Travel": travel_type,
    "Class": travel_class,
    "Flight Distance": flight_distance,
    "Inflight wifi service": wifi,
    "Departure/Arrival time convenient": time_conv,
    "Ease of Online booking": booking,
    "Gate location": gate,
    "Food and drink": food,
    "Online boarding": boarding,
    "Seat comfort": seat,
    "Inflight entertainment": entertainment,
    "On-board service": onboard,
    "Leg room service": legroom,
    "Baggage handling": baggage,
    "Checkin service": checkin,
    "Inflight service": inflight,
    "Cleanliness": cleanliness,
    "Departure Delay in Minutes": dep_delay,
    "Arrival Delay in Minutes": arr_delay
}])

# --- PREDICT ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("Result:")
    st.write("✅ Satisfied" if prediction == 1 else "❌ Not satisfied")
    st.write(f"Probability: {proba:.2f}")

    # --- SHAP ---
    st.subheader("SHAP Explanation")

    preprocessor = model.named_steps["preprocessor"]
    model_inner = model.named_steps["model"]

    X_transformed = preprocessor.transform(input_data)

    explainer = shap.TreeExplainer(model_inner)
    shap_values = explainer.shap_values(X_transformed)

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_transformed[0]
        )
    )

    st.pyplot(bbox_inches="tight")

import streamlit as st
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

st.set_page_config(page_title="Airline Satisfaction", layout="wide")

st.title("✈️ Airline Passenger Satisfaction")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# ---------- PREP ----------
y = df['satisfaction'].map({
    'neutral or dissatisfied': 0,
    'satisfied': 1
})

X = df.drop(columns=['satisfaction'])

cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

# ---------- PIPELINE ----------
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric='logloss'
    ))
])

# ---------- TRAIN ----------
@st.cache_resource
def train_model():
    model_pipeline.fit(X, y)
    return model_pipeline

model = train_model()

st.success("Model trained successfully!")

# ---------- UI ----------
st.subheader("Enter passenger details")

# ----------- BASIC INFO -----------
st.markdown("### ✈️ Passenger Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])

with col2:
    age = st.slider("Age", 10, 80, 30)
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])

with col3:
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    flight_distance = st.number_input("Flight Distance", 0, 5000, 500)


# ----------- SERVICE RATINGS -----------
st.markdown("### ⭐ Service Ratings")

col1, col2, col3 = st.columns(3)

with col1:
    wifi = st.slider("Wifi", 0, 5, 3)
    booking = st.slider("Booking", 0, 5, 3)
    gate = st.slider("Gate", 0, 5, 3)
    boarding = st.slider("Boarding", 0, 5, 3)

with col2:
    seat = st.slider("Seat", 0, 5, 3)
    entertainment = st.slider("Entertainment", 0, 5, 3)
    onboard = st.slider("On-board", 0, 5, 3)
    legroom = st.slider("Legroom", 0, 5, 3)

with col3:
    baggage = st.slider("Baggage", 0, 5, 3)
    checkin = st.slider("Check-in", 0, 5, 3)
    inflight = st.slider("Inflight", 0, 5, 3)
    cleanliness = st.slider("Cleanliness", 0, 5, 3)


# ----------- EXTRA -----------
st.markdown("### ⏱ Travel Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    time_conv = st.slider("Time convenient", 0, 5, 3)

with col2:
    food = st.slider("Food", 0, 5, 3)

with col3:
    dep_delay = st.number_input("Departure Delay", 0, 1000, 0)
    arr_delay = st.number_input("Arrival Delay", 0, 1000, 0)


# ---------- INPUT ----------
input_data = pd.DataFrame(columns=X.columns)
input_data.loc[0] = 0

input_data["Gender"] = gender
input_data["Customer Type"] = customer_type
input_data["Age"] = age
input_data["Type of Travel"] = travel_type
input_data["Class"] = travel_class
input_data["Flight Distance"] = flight_distance

input_data["Inflight wifi service"] = wifi
input_data["Departure/Arrival time convenient"] = time_conv
input_data["Ease of Online booking"] = booking
input_data["Gate location"] = gate
input_data["Food and drink"] = food
input_data["Online boarding"] = boarding
input_data["Seat comfort"] = seat
input_data["Inflight entertainment"] = entertainment
input_data["On-board service"] = onboard
input_data["Leg room service"] = legroom
input_data["Baggage handling"] = baggage
input_data["Checkin service"] = checkin
input_data["Inflight service"] = inflight
input_data["Cleanliness"] = cleanliness

input_data["Departure Delay in Minutes"] = dep_delay
input_data["Arrival Delay in Minutes"] = arr_delay


# ---------- PREDICT ----------
if st.button("Predict"):

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction")

    if pred == 1:
        st.success(f"✅ Satisfied (Probability: {proba:.2f})")
    else:
        st.error(f"❌ Not satisfied (Probability: {proba:.2f})")

    st.progress(float(proba))

    # ---------- SHAP ----------
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

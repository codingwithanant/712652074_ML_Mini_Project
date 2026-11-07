
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "road_accident_severity_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

st.title("🚗 Road Accident Severity Prediction")
st.write(
    "Enter the recorded accident details below to estimate its severity."
)
st.info(
    "This tool estimates the severity of a recorded accident. "
    "It does not predict whether an accident will occur."
)

def dropdown(label, options, key):
    return st.selectbox(label, options, key=key)

with st.form("accident_form"):
    st.subheader("1. Time and driver details")
    c1, c2, c3 = st.columns(3)

    with c1:
        time = dropdown("Time", [
            "08:00:00", "09:00:00", "10:00:00",
            "11:00:00", "12:00:00", "13:00:00",
            "14:00:00", "15:00:00", "16:00:00",
            "17:00:00", "18:00:00", "19:00:00",
            "20:00:00", "21:00:00", "22:00:00"
        ], "time")

        day = dropdown("Day of week", [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ], "day")

        age = dropdown("Driver age band", [
            "Under 18", "18-30", "31-50", "Over 51"
        ], "age")

    with c2:
        sex = dropdown("Driver sex", ["Male", "Female"], "sex")

        education = dropdown("Educational level", [
            "Above high school", "Elementary school", "High school",
            "Illiterate", "Junior high school", "Writing & reading"
        ], "education")

        relation = dropdown("Driver-vehicle relation", [
            "Employee", "Other", "Owner"
        ], "relation")

    with c3:
        experience = dropdown("Driving experience", [
            "1-2yr", "2-5yr", "5-10yr",
            "Above 10yr", "Below 1yr", "No Licence"
        ], "experience")

        vehicle = dropdown("Type of vehicle", [
            "Automobile", "Bajaj", "Bicycle", "Long lorry",
            "Lorry (11?40Q)", "Lorry (41?100Q)", "Motorcycle",
            "Other", "Pick up upto 10Q", "Public (12 seats)",
            "Public (13?45 seats)", "Public (> 45 seats)",
            "Ridden horse", "Special vehicle", "Stationwagen",
            "Taxi", "Turbo"
        ], "vehicle")

        owner = dropdown("Vehicle owner", [
            "Governmental", "Organization", "Other", "Owner"
        ], "owner")

    st.subheader("2. Vehicle and road details")
    c1, c2, c3 = st.columns(3)

    with c1:
        service_year = dropdown("Service year of vehicle", [
            "1-2yr", "2-5yrs", "5-10yrs", "Above 10yr", "Below 1yr"
        ], "service_year")

        defect = dropdown("Vehicle defect", [
            "5", "7", "No defect"
        ], "defect")

        area = dropdown("Accident area", [
            "Church areas", "Hospital areas", "Industrial areas",
            "Market areas", "Office areas", "Other",
            "Outside rural areas", "Recreational areas",
            "Residential areas", "Rural village areas",
            "Rural village areasOffice areas", "School areas"
        ], "area")

        lanes = dropdown("Lanes or medians", [
            "Double carriageway (median)", "One way",
            "Two-way (divided with broken lines road marking)",
            "Two-way (divided with solid lines road marking)",
            "Undivided Two way", "other"
        ], "lanes")

    with c2:
        alignment = dropdown("Road alignment", [
            "Escarpments", "Gentle horizontal curve", "Sharp reverse curve",
            "Steep grade downward with mountainous terrain",
            "Steep grade upward with mountainous terrain",
            "Tangent road with flat terrain",
            "Tangent road with mild grade and flat terrain",
            "Tangent road with mountainous terrain and",
            "Tangent road with rolling terrain"
        ], "alignment")

        junction = dropdown("Type of junction", [
            "Crossing", "No junction", "O Shape", "Other",
            "T Shape", "X Shape", "Y Shape"
        ], "junction")

        surface_type = dropdown("Road surface type", [
            "Asphalt roads", "Asphalt roads with some distress",
            "Earth roads", "Gravel roads", "Other"
        ], "surface_type")

        surface_condition = dropdown("Road surface condition", [
            "Dry", "Flood over 3cm. deep", "Snow", "Wet or damp"
        ], "surface_condition")

    with c3:
        light = dropdown("Light conditions", [
            "Darkness - lights lit", "Darkness - lights unlit",
            "Darkness - no lighting", "Daylight"
        ], "light")

        weather = dropdown("Weather conditions", [
            "Cloudy", "Fog or mist", "Normal", "Other",
            "Raining", "Raining and Windy", "Snow", "Windy"
        ], "weather")

        vehicles = st.selectbox(
            "Number of vehicles involved",
            [1, 2, 3, 4, 6, 7],
            index=1
        )

        movement = dropdown("Vehicle movement", [
            "Entering a junction", "Getting off", "Going straight",
            "Moving Backward", "Other", "Overtaking", "Parked",
            "Reversing", "Stopping", "Turnover", "U-Turn",
            "Waiting to go"
        ], "movement")

    submitted = st.form_submit_button(
        "🔍 Predict Severity", use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Time": time,
        "Day_of_week": day,
        "Age_band_of_driver": age,
        "Sex_of_driver": sex,
        "Educational_level": education,
        "Vehicle_driver_relation": relation,
        "Driving_experience": experience,
        "Type_of_vehicle": vehicle,
        "Owner_of_vehicle": owner,
        "Service_year_of_vehicle": service_year,
        "Defect_of_vehicle": defect,
        "Area_accident_occured": area,
        "Lanes_or_Medians": lanes,
        "Road_allignment": alignment,
        "Types_of_Junction": junction,
        "Road_surface_type": surface_type,
        "Road_surface_conditions": surface_condition,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Number_of_vehicles_involved": vehicles,
        "Vehicle_movement": movement
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.write("Debug — input sent to model:")
        st.dataframe(input_data)
        st.subheader("Prediction result")

        if str(prediction).lower() == "fatal injury":
            st.error(f"Estimated severity: {prediction}")
        elif str(prediction).lower() == "serious injury":
            st.warning(f"Estimated severity: {prediction}")
        else:
            st.success(f"Estimated severity: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.caption("Model probability estimates (not guaranteed outcomes)")
            prob_df = pd.DataFrame({
                "Severity": model.classes_,
                "Estimated probability (%)": probabilities * 100
            })
            st.bar_chart(
                prob_df.set_index("Severity")["Estimated probability (%)"]
            )
        else:
            st.caption(
                "This SVM model does not provide probability estimates "
                "because probability=True was not enabled during training."
            )

        with st.expander("View submitted inputs"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the model file matches the trained notebook "
            "and that input categories match the dataset."
        )


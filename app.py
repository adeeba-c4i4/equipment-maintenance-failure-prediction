import streamlit as st
import pandas as pd
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Equipment Failure Prediction",
    layout="centered"
)

# --- Define Artifacts Directory (Using your specific path) ---
ARTIFACTS_DIR = r"D:\projects\equipment_failure_prediction\artifacts"

# --- Load Logistic Model and Encoder ---
@st.cache_resource
def load_artifacts():
    """Loads the logistic regression model and the label encoder."""
    try:
        model_path = os.path.join(ARTIFACTS_DIR, "logistic_regression_model.pkl")
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: Could not find 'logistic_regression_model.pkl' in '{ARTIFACTS_DIR}'")
        st.error("Please run the `full_analysis_and_training.py` script first.")
        return None, None

    try:
        le_path = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')
        le = joblib.load(le_path)
    except FileNotFoundError:
        st.error(f"Error: Could not find 'label_encoder.pkl' in '{ARTIFACTS_DIR}'")
        st.error("Please run the `full_analysis_and_training.py` script first.")
        return None, None
        
    return model, le

model, le = load_artifacts()

# --- Feature list ---
features = [
    'process_errors', 'sensor_5', 'sensor_8', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'age_of_equipment'
]

# --- App Title ---
st.title("Equipment Failure Prediction")

if not model or not le:
    st.stop() # Stops the app if artifacts didn't load

# --- Input Form (in the middle) ---
input_data = {}

with st.form(key="input_form"):
    st.subheader("Input Equipment Feature Values")

    # Create a 3-column layout for inputs
    cols = st.columns(3)

    for i, feature in enumerate(features):
        col = cols[i % 3] # This creates the 3x3 grid
        input_data[feature] = col.number_input(
            feature.replace('_', ' ').capitalize(),
            value=0.0,
            format="%.4f"
        )

    # Submit button inside the form
    # FIX: Replaced use_container_width=True with width='stretch'
    run_button = st.form_submit_button("Predict Failure", type="primary", width='stretch')


# --- Main Output Section ---
st.header("Prediction Result")
output_container = st.container(border=True)

if run_button:
    # This block runs ONLY when the "Predict Failure" button is clicked
    input_df = pd.DataFrame([input_data])

    try:
        pred_encoded = model.predict(input_df)
        pred_proba = model.predict_proba(input_df)
        pred_label = le.inverse_transform(pred_encoded)[0]

        with output_container:
            st.subheader("Model Used: Logistic Regression")

            # Use st.metric for a clear, visual output
            if pred_label.lower() == 'failure':
                st.metric(
                    label="Model Prediction",
                    value=pred_label.capitalize(),
                    delta="High Risk",
                    delta_color="inverse"
                )
                st.error(f"The model predicts a high probability of **{pred_label.capitalize()}**.")
            else:
                st.metric(
                    label="Model Prediction",
                    value=pred_label.capitalize(),
                    delta="Low Risk",
                    delta_color="off"
                )
                st.success(f"The model predicts **{pred_label.capitalize()}**.")

            st.divider()

            st.subheader("Prediction Confidence")
            
            # Create a DataFrame for the bar chart
            proba_df_chart = pd.DataFrame({
                "Probability": pred_proba[0]
            }, index=le.classes_)
            
            st.bar_chart(proba_df_chart)

            # Optional: Show the exact percentages in a table
            st.write("Probabilities:")
            proba_df_table = proba_df_chart.copy()
            proba_df_table["Probability"] = proba_df_table["Probability"].apply(lambda x: f"{x:.1%}")
            # FIX: Replaced use_container_width=True with width='stretch'
            st.dataframe(proba_df_table, width='stretch')


    except Exception as e:
        output_container.error(f"Error during prediction: {e}")

else:
    # This is the default message before the button is pressed
    output_container.info('Enter feature values above and click "Predict Failure".')
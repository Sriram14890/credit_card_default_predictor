import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier


# =========================================================
# LOAD SAVED MODEL
# =========================================================

preprocessor = joblib.load(
    "credit_default_preprocessor.pkl"
)

xgb_model = XGBClassifier()

xgb_model.load_model(
    "credit_default_xgb.json"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Default Prediction")

st.write(
    "Enter the customer's information below to predict "
    "the probability of credit card default."
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    limit_bal = st.number_input(
        "Credit Limit",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )


with col2:

    sex = st.selectbox(
        "Sex",
        [1, 2],
        format_func=lambda x:
        "Male" if x == 1 else "Female"
    )

    education = st.selectbox(
        "Education",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Others"
        }[x]
    )

    marriage = st.selectbox(
        "Marriage",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "Married",
            2: "Single",
            3: "Others"
        }[x]
    )


# =========================================================
# REPAYMENT HISTORY
# =========================================================

st.header("Repayment History")

st.write(
    "Enter the repayment status for the previous months."
)

col1, col2, col3 = st.columns(3)

with col1:

    pay_0 = st.number_input(
        "PAY_0",
        min_value=-2,
        max_value=9,
        value=0
    )

    pay_2 = st.number_input(
        "PAY_2",
        min_value=-2,
        max_value=9,
        value=0
    )


with col2:

    pay_3 = st.number_input(
        "PAY_3",
        min_value=-2,
        max_value=9,
        value=0
    )

    pay_4 = st.number_input(
        "PAY_4",
        min_value=-2,
        max_value=9,
        value=0
    )


with col3:

    pay_5 = st.number_input(
        "PAY_5",
        min_value=-2,
        max_value=9,
        value=0
    )

    pay_6 = st.number_input(
        "PAY_6",
        min_value=-2,
        max_value=9,
        value=0
    )


# =========================================================
# BILL AMOUNTS
# =========================================================

st.header("Bill Amounts")

col1, col2, col3 = st.columns(3)

with col1:

    bill_amt1 = st.number_input(
        "BILL_AMT1",
        value=50000.0
    )

    bill_amt2 = st.number_input(
        "BILL_AMT2",
        value=50000.0
    )


with col2:

    bill_amt3 = st.number_input(
        "BILL_AMT3",
        value=50000.0
    )

    bill_amt4 = st.number_input(
        "BILL_AMT4",
        value=50000.0
    )


with col3:

    bill_amt5 = st.number_input(
        "BILL_AMT5",
        value=50000.0
    )

    bill_amt6 = st.number_input(
        "BILL_AMT6",
        value=50000.0
    )


# =========================================================
# PAYMENT AMOUNTS
# =========================================================

st.header("Previous Payment Amounts")

col1, col2, col3 = st.columns(3)

with col1:

    pay_amt1 = st.number_input(
        "PAY_AMT1",
        min_value=0.0,
        value=2000.0
    )

    pay_amt2 = st.number_input(
        "PAY_AMT2",
        min_value=0.0,
        value=2000.0
    )


with col2:

    pay_amt3 = st.number_input(
        "PAY_AMT3",
        min_value=0.0,
        value=2000.0
    )

    pay_amt4 = st.number_input(
        "PAY_AMT4",
        min_value=0.0,
        value=2000.0
    )


with col3:

    pay_amt5 = st.number_input(
        "PAY_AMT5",
        min_value=0.0,
        value=2000.0
    )

    pay_amt6 = st.number_input(
        "PAY_AMT6",
        min_value=0.0,
        value=2000.0
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "Predict Default",
    type="primary"
):

    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------

    total_payment = (
        pay_amt1
        + pay_amt2
        + pay_amt3
        + pay_amt4
        + pay_amt5
        + pay_amt6
    )

    avg_payment = total_payment / 6

    avg_bill = (
        bill_amt1
        + bill_amt2
        + bill_amt3
        + bill_amt4
        + bill_amt5
        + bill_amt6
    ) / 6

    num_delays = sum([
        pay_0 > 0,
        pay_2 > 0,
        pay_3 > 0,
        pay_4 > 0,
        pay_5 > 0,
        pay_6 > 0
    ])

    max_delay = max(
        0,
        pay_0,
        pay_2,
        pay_3,
        pay_4,
        pay_5,
        pay_6
    )


    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "limit_bal": limit_bal,

        "sex": sex,

        "education": education,

        "marriage": marriage,

        "age": age,

        "pay_0": pay_0,

        "pay_2": pay_2,

        "pay_3": pay_3,

        "pay_4": pay_4,

        "pay_5": pay_5,

        "pay_6": pay_6,

        "bill_amt1": bill_amt1,

        "bill_amt2": bill_amt2,

        "bill_amt3": bill_amt3,

        "bill_amt4": bill_amt4,

        "bill_amt5": bill_amt5,

        "bill_amt6": bill_amt6,

        "pay_amt1": pay_amt1,

        "pay_amt2": pay_amt2,

        "pay_amt3": pay_amt3,

        "pay_amt4": pay_amt4,

        "pay_amt5": pay_amt5,

        "pay_amt6": pay_amt6,

        "total_payment": total_payment,

        "avg_payment": avg_payment,

        "avg_bill": avg_bill,

        "num_delays": num_delays,

        "max_delay": max_delay

    }])


    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    input_transformed = preprocessor.transform(
        input_data
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    probability = xgb_model.predict_proba(
        input_transformed
    )[0][1]


    # Your selected threshold
    threshold = 0.50

    prediction = int(
        probability >= threshold
    )


    # -----------------------------------------------------
    # Display Result
    # -----------------------------------------------------

    st.divider()

    if prediction == 1:

        st.error(
            "⚠️ High Risk: Customer is predicted to default."
        )

    else:

        st.success(
            "✅ Low Risk: Customer is predicted not to default."
        )


    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )
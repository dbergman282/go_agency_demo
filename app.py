import streamlit as st
import pandas as pd
from model_utils import train_model, predict_customers
from ai_assistant import ask_ai

# Load data
data = pd.read_csv("data/marketing_campaign.csv")

st.title("Predict Customer Purchases with AI")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page:", [
    "Feature Selection",
    "Model Training",
    "Customer Comparison",
    "Ask the AI"
])

# Store session state
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "model" not in st.session_state:
    st.session_state.model = None

if page == "Feature Selection":
    st.header("Select Features You Think Are Important")
    features = [col for col in data.columns if col not in ["ID", "Response"]]  # Drop ID and Target
    selected = st.multiselect("Pick your features:", features)
    if st.button("Save Feature Selection"):
        st.session_state.selected_features = selected
        st.success("Features saved! Now go to Model Training.")

elif page == "Model Training":
    st.header("Train Model Based on Selected Features")
    if not st.session_state.selected_features:
        st.warning("Please select features first!")
    else:
        model, report = train_model(data, st.session_state.selected_features)
        st.session_state.model = model
        st.subheader("Model Report:")
        st.json(report)

elif page == "Customer Comparison":
    st.header("Compare Two Customers")
    if st.session_state.model is None:
        st.warning("Please train the model first!")
    else:
        customer_idx1 = st.number_input("Select Customer A Row Number", min_value=0, max_value=len(data)-1, value=0)
        customer_idx2 = st.number_input("Select Customer B Row Number", min_value=0, max_value=len(data)-1, value=1)
        pred1, pred2 = predict_customers(data, customer_idx1, customer_idx2, st.session_state)
        st.write(f"**Customer A Purchase Probability:** {pred1:.2f}")
        st.write(f"**Customer B Purchase Probability:** {pred2:.2f}")

elif page == "Ask the AI":
    st.header("Ask the AI about the model")
    if st.session_state.model is None:
        st.warning("Please train the model first!")
    else:
        user_question = st.text_input("Your Question:")
        if user_question:
            answer = ask_ai(user_question, st.session_state)
            st.write("### AI Answer:")
            st.write(answer)

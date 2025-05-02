import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import train_model, predict_customers
from ai_assistant import ask_ai

# Load data
data = pd.read_csv("data/marketing_campaign.csv")

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("Predict Customer Purchases with AI")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page:", [
    "Data Overview",
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

# Data Overview Page
if page == "Data Overview":
    st.header("📊 Data Overview")
    st.markdown("""
    This app uses a real-world marketing campaign dataset to predict whether a customer will respond to a future campaign.
    
    **Target Variable:**  
    - `Response` (0 = No, 1 = Yes): Did the customer respond to the most recent marketing campaign?

    **Source:**  
    This dataset is based on the ["Customer Personality Analysis"](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) dataset from Kaggle, originally released by a Portuguese retail company.

    ---
    """)

    if st.checkbox("🔍 Show sample data (first 20 rows)"):
        st.dataframe(data.head(20))

    if st.checkbox("📚 Show column descriptions"):
        st.markdown("""
        **Demographic & Household Info:**
        - `Income`: Yearly income of the customer.
        - `Kidhome`: Number of children in the household.
        - `Teenhome`: Number of teenagers in the household.
        - `Age`: Customer age (engineered).
        - `Customer_Days`: Days since enrollment.

        **Purchase Behavior:**
        - `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`: Spending on product categories.
        - `MntTotal`: Total spending.
        - `MntRegularProds`: Spending excluding gold products.

        **Campaign Interaction:**
        - `AcceptedCmp1`–`AcceptedCmp5`: Responses to previous campaigns.
        - `AcceptedCmpOverall`: Total campaigns accepted.
        - `Response`: Target — response to the last campaign.

        **Engagement Metrics:**
        - `Recency`: Days since last purchase.
        - `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`: Purchase channels.
        - `NumWebVisitsMonth`: Website visits in the last month.

        **Other:**
        - `Complain`: Whether the customer complained.
        - `Z_CostContact`, `Z_Revenue`: Business constants.

        **One-Hot Encoded:**
        - `marital_*`: Marital status.
        - `education_*`: Education level.
        """)

    st.markdown("---")
    st.subheader("📈 Campaign Response Rate")
    response_rate = data["Response"].value_counts(normalize=True).sort_index()
    fig1, ax1 = plt.subplots()
    ax1.bar(["No", "Yes"], response_rate.values, color=["skyblue", "orange"])
    ax1.set_ylabel("Proportion")
    ax1.set_title("Customer Response to Last Campaign")
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("💸 Spending vs. Age")
    fig2, ax2 = plt.subplots()
    ax2.scatter(data["Age"], data["MntTotal"], alpha=0.5)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Total Spending (MntTotal)")
    ax2.set_title("Customer Spending vs. Age")
    st.pyplot(fig2)

# Feature Selection Page
elif page == "Feature Selection":
    st.header("🔍 Select Features You Think Are Important")
    features = [col for col in data.columns if col not in ["ID", "Response"]]
    selected = st.multiselect("Pick your features:", features)
    if st.button("Save Feature Selection"):
        st.session_state.selected_features = selected
        st.success("✅ Features saved! Now go to Model Training.")

# Model Training Page
elif page == "Model Training":
    st.header("🏋️ Train Model Based on Selected Features")
    if not st.session_state.selected_features:
        st.warning("Please select features first!")
    else:
        model, report = train_model(data, st.session_state.selected_features)
        st.session_state.model = model
        st.subheader("🧠 Model Report:")
        st.json(report)

# Customer Comparison Page
elif page == "Customer Comparison":
    st.header("🧑‍🤝‍🧑 Compare Two Customers")
    if st.session_state.model is None:
        st.warning("Please train the model first!")
    else:
        customer_idx1 = st.number_input("Select Customer A Row Number", min_value=0, max_value=len(data)-1, value=0)
        customer_idx2 = st.number_input("Select Customer B Row Number", min_value=0, max_value=len(data)-1, value=1)
        pred1, pred2 = predict_customers(data, customer_idx1, customer_idx2, st.session_state)
        st.write(f"**Customer A Purchase Probability:** {pred1:.2f}")
        st.write(f"**Customer B Purchase Probability:** {pred2:.2f}")

# Ask the AI Page
elif page == "Ask the AI":
    st.header("💬 Ask the AI about the Model")
    if st.session_state.model is None:
        st.warning("Please train the model first!")
    else:
        user_question = st.text_input("Your Question:")
        if user_question:
            answer = ask_ai(user_question, st.session_state)
            st.write("### AI Answer:")
            st.write(answer)

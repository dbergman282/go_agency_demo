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
        with st.spinner("Training model..."):
            model, report, X_test, y_test, feature_names = train_model(data, st.session_state.selected_features)
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_names = feature_names

        st.success("✅ Model training complete!")

        # Format and show metrics table
        st.subheader("📋 Classification Metrics")
        import numpy as np
        report_df = pd.DataFrame(report).transpose()
        display_metrics = report_df.loc[["0", "1", "accuracy", "macro avg", "weighted avg"]]
        clean_metrics = display_metrics[["precision", "recall", "f1-score", "support"]].round(2)
        st.dataframe(clean_metrics)

        # Bar plot: Precision & Recall for Class 0 and 1
        st.subheader("📊 Precision vs. Recall (by Class)")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        labels = ["Did Not Respond (0)", "Responded (1)"]
        x = np.arange(len(labels))
        width = 0.35
        precision = display_metrics.loc[["0", "1"], "precision"].values
        recall = display_metrics.loc[["0", "1"], "recall"].values
        ax.bar(x - width/2, precision, width, label='Precision', color='skyblue')
        ax.bar(x + width/2, recall, width, label='Recall', color='orange')
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Model Precision and Recall by Class")
        ax.legend()
        st.pyplot(fig)

        # Natural language interpretation
        st.subheader("🧠 Model Interpretation")
        precision1 = report_df.loc["1", "precision"]
        recall1 = report_df.loc["1", "recall"]
        f1_1 = report_df.loc["1", "f1-score"]
        st.markdown(f"""
        - For customers who **responded** to the campaign:
            - Precision: **{precision1:.2f}**
            - Recall: **{recall1:.2f}**
            - F1-Score: **{f1_1:.2f}**
        """)

        # ROC Curve
        st.subheader("📉 ROC Curve")
        from sklearn.metrics import roc_curve, auc
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC)")
        ax_roc.legend()
        st.pyplot(fig_roc)

        # Feature importance
        st.subheader("🔍 Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        ax_imp.barh(feat_df["Feature"][:10][::-1], feat_df["Importance"][:10][::-1], color="slateblue")
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Top 10 Important Features")
        st.pyplot(fig_imp)


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

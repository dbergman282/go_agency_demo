import openai
import os

# Make sure you have your OpenAI key set as an environment variable or manually
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")  # <-- Replace if needed

def ask_ai(user_question, session_state):
    model = session_state.model
    selected_features = session_state.selected_features

    # Basic context you give to the AI
    context = f"""
    We built a model to predict customer purchase behavior based on these features: {selected_features}.
    The model is a basic XGBoost classifier. The goal is to predict if a customer will respond positively.
    """

    prompt = context + "\n\nUser question: " + user_question + "\n\nAnswer in clear, friendly language."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change if you have GPT-4 access
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that explains machine learning models to marketing executives."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        return response['choices'][0]['message']['content']
    
    except Exception as e:
        return f"Error querying AI: {e}"

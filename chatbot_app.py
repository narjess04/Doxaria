# chatbot_app.py
import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Charger le modèle fine-tuné et le tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-medical")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-medical")

# Générateur de texte
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Fonction de réponse du chatbot avec le modèle fine-tuné
def chatbot_response(user_input):
    prompt = f"Question: {user_input}\nAnswer:"
    output = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].strip()

# Interface utilisateur Streamlit
st.title("ChatBot Médical")

st.write("Bienvenue! Posez une question au chatbot:")

# Champ de texte pour la question de l'utilisateur
user_input = st.text_input("Votre question", "")

# Lorsque l'utilisateur soumet une question
if st.button('Envoyer'):
    if user_input:
        bot_response = chatbot_response(user_input)
        st.write(f"**ChatBot :** {bot_response}")
    else:
        st.write("Veuillez entrer une question.")

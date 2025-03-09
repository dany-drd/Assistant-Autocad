import streamlit as st
import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  
MISTRAL_API_KEY = "TA_CLE_API"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Chargement FAISS
def load_faiss_and_metadata():
    index_path = os.path.join(SAVE_DIR, "faiss_index.idx")
    metadata_path = os.path.join(SAVE_DIR, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

# Recherche dans FAISS
def search_faiss(query, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])
    index, metadata = load_faiss_and_metadata()
    
    if index is None:
        return []

    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = [metadata[str(i)] for i in indices[0] if str(i) in metadata]
    return results

# Fonction pour interagir avec Mistral et conserver le contexte
def query_mistral(messages):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "mistral-medium", "messages": messages}

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Erreur API Mistral : {response.text}"

# Interface Web Streamlit
st.set_page_config(page_title="Assistant RE2020", page_icon="⚡")
st.title("🔍 Assistant RE2020 avec Mistral AI")
st.write("Posez une question sur la réglementation environnementale RE2020.")

# Initialiser l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Tu es un assistant spécialisé en RE2020."}]

# Afficher l'historique
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.text_area("Vous :", value=msg["content"], height=75, disabled=True)
    elif msg["role"] == "assistant":
        st.text_area("Assistant :", value=msg["content"], height=75, disabled=True)

query = st.text_input("📝 Entrez votre question :", placeholder="Ex: Quels sont les objectifs de la RE2020 ?")

if st.button("🔎 Rechercher"):
    if query:
        with st.spinner("Recherche en cours... ⏳"):
            passages = search_faiss(query)
            context = "\n".join(passages) if passages else "Aucun passage trouvé."
            
            # Ajouter la question à l'historique
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Envoyer la conversation complète à Mistral
            response = query_mistral(st.session_state.messages)
            
            # Ajouter la réponse de Mistral à l'historique
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Afficher la réponse
            st.subheader("📌 Réponse :")
            st.write(response)
    else:
        st.warning("⚠ Veuillez entrer une question avant de rechercher.")

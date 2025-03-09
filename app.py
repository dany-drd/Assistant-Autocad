import streamlit as st
import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # Utilisation du dossier du script
MISTRAL_API_KEY = "1ynaJUIWuhjOytyTommUH1f19L3Mf2t9"  # Mets ta vraie clé API
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Chargement du modèle SentenceTransformer (optimisé : ne charge qu'une fois)
model = SentenceTransformer(MODEL_NAME)

# Fonction pour charger FAISS et les métadonnées
def load_faiss_and_metadata():
    index_path = os.path.join(SAVE_DIR, "faiss_index.idx")
    metadata_path = os.path.join(SAVE_DIR, "metadata.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Le fichier d'index FAISS est introuvable : {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Le fichier metadata.json est introuvable : {metadata_path}")

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

# Recherche dans FAISS
def search_faiss(query, top_k=3):
    query_embedding = model.encode([query])  # Retourne directement un tableau NumPy
    index, metadata = load_faiss_and_metadata()
    distances, indices = index.search(query_embedding, top_k)
    results = [metadata[str(i)] for i in indices[0] if str(i) in metadata]
    return results

# Appel API Mistral
def query_mistral(query, passages):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    context = "\n".join(passages)

    prompt = [
        {
            "role": "system",
            "content": (
                "Tu es un expert en AutoCAD, spécialisé dans l'utilisation et les fonctionnalités du logiciel. "
                "Répond toujours en français"
                "Ta mission est de répondre aux questions des utilisateurs en t'appuyant sur les informations disponibles dans le manuel AutoCAD. "
                "Tu peux prendre certaines libertés dans l'explication pour la rendre plus claire et pédagogique, mais tu dois rester fidèle aux documents fournis. "
                "Si une information n'est pas explicitement mentionnée dans les documents, tu peux fournir une interprétation raisonnable en précisant qu'il s'agit d'une extrapolation. "
                "Si une question ne concerne pas AutoCAD ou si l'information n'est pas disponible dans le contexte fourni, explique poliment que tu es spécialisé dans AutoCAD "
                "et invite l'utilisateur à poser des questions sur ce logiciel."
            )
        },
        {
            "role": "user",
            "content": f"Contexte du manuel AutoCAD :\n{context}\n\nQuestion : {query}"
        }
    ]

    data = {"model": "mistral-medium", "messages": prompt, "temperature": 0.5}

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            return "Réponse invalide de l'API Mistral."
    else:
        return f"Erreur API Mistral : {response.text}"

# Interface Web Streamlit
st.set_page_config(page_title="Assistant AutoCAD", page_icon="🔧")
st.title("🔧 Assistant AutoCAD")
st.write("Posez une question sur AutoCAD et obtenez une réponse instantanée")

query = st.text_input("📝 Entrez votre question :", placeholder="Quelles sont les principales commandes AutoCAD ?")

if st.button("🔎 Rechercher"):
    if query:
        with st.spinner("Recherche en cours... ⏳"):
            try:
                passages = search_faiss(query)
                response = query_mistral(query, passages) if passages else "Aucun passage pertinent trouvé."
            except FileNotFoundError as e:
                response = f"❌ Erreur : {str(e)}"
        st.subheader("📌 Réponse :")
        st.write(response)
    else:
        st.warning("⚠️ Veuillez entrer une question avant de rechercher.")

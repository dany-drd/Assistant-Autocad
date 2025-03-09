import os
import json
import faiss
import numpy as np
import requests
import streamlit as st
from fpdf import FPDF
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MISTRAL_API_KEY = "VOTRE_CLE_API"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Fonction pour charger FAISS et les métadonnées
def load_faiss_and_metadata():
    index_path = os.path.join(SAVE_DIR, "faiss_index.idx")
    metadata_path = os.path.join(SAVE_DIR, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("L'index FAISS ou le fichier metadata.json est introuvable.")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

# Recherche dans FAISS
def search_faiss(query, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])
    index, metadata = load_faiss_and_metadata()
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    
    results = [metadata[str(i)] for i in indices[0] if str(i) in metadata]
    return results

# Fonction pour structurer la réponse et la transformer en rapport
def query_mistral(query, passages):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    context = "\n".join(passages)

    prompt = f"""
    Tu es un assistant spécialisé dans la rédaction de rapports détaillés.
    
    Voici des extraits de documents pertinents pour répondre à la question :
    
    {context}
    
    Rédige un document structuré avec :
    
    - Introduction
    - Développement détaillé
    - Conclusion synthétique
    
    Question utilisateur : {query}
    
    Réponds en français de manière fluide et bien organisée.
    """

    data = {"model": "mistral-medium", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Erreur API Mistral : {response.text}"

# Fonction pour générer un PDF
def generate_pdf(text, filename="rapport.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    return filename

# Interface Streamlit
st.title("🔍 Assistant RE2020 - Génération de documents")
query = st.text_area("📝 Décrivez votre besoin :", placeholder="Ex: Rédigez un rapport sur l'impact énergétique des bâtiments…")

if st.button("📝 Générer un rapport"):
    if query:
        with st.spinner("Génération du document en cours... ⏳"):
            passages = search_faiss(query)
            response = query_mistral(query, passages) if passages else "Aucun résultat trouvé."
        
        # Affichage du texte généré
        st.subheader("📌 Document généré :")
        st.write(response)

        # Génération du PDF et bouton de téléchargement
        pdf_filename = generate_pdf(response)
        with open(pdf_filename, "rb") as f:
            st.download_button("📥 Télécharger le rapport en PDF", f, file_name="rapport.pdf", mime="application/pdf")
    else:
        st.warning("⚠ Veuillez entrer une question avant de générer le document.")

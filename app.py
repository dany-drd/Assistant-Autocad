import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MISTRAL_API_KEY = "VOTRE_CLE_API"  # Remplace par ta clé
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

# Fonction pour structurer la réponse
def query_mistral(query, passages):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    context = "\n".join(passages)

    prompt = f"""
    Tu es un assistant expert en rédaction de rapports.
    Voici des extraits de documents pertinents pour répondre à la question :
    
    {context}
    
    Rédige un document bien structuré en intégrant les informations de manière fluide.
    Utilise la structure suivante :
    
    - Introduction
    - Développement détaillé
    - Conclusion claire et synthétique
    
    Question utilisateur : {query}
    
    Réponds de manière complète et cohérente en français.
    """

    data = {"model": "mistral-medium", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Erreur API Mistral : {response.text}"

if __name__ == "__main__":
    query = input("Posez votre question : ")
    passages = search_faiss(query)
    response = query_mistral(query, passages) if passages else "Aucun résultat pertinent trouvé."
    
    print("\n📌 Document généré :\n")
    print(response)

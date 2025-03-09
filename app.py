def query_mistral(query, passages):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    context = "\n".join(passages)
    
    prompt = f"""Tu es un assistant expert qui rédige des rapports détaillés en utilisant les documents suivants :
    
    {context}
    
    Rédige une réponse sous forme de document structuré, en intégrant les éléments pertinents sans simple copier-coller. 
    Assure-toi d’utiliser un ton professionnel et une structure logique :
    
    - Introduction
    - Développement (avec explications détaillées)
    - Conclusion
    
    Question utilisateur : {query}
    
    Réponds en français sous forme d’un texte fluide et bien structuré.
    """
    
    data = {"model": "mistral-medium", "messages": [{"role": "user", "content": prompt}]}
    
    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Erreur API Mistral : {response.text}"

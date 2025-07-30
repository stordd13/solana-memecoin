# %%
import openai
import numpy as np
from typing import Tuple

EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

def get_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """ 
    Calcule la similarité cosinus entre deux embeddings avec normalisation explicite.
    
    Returns:
    float: Similarité cosinus normalisée entre -1 et 1
    """
    # Normalisation explicite des embeddings
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Éviter la division par zéro
    
    # Calcul de la similarité cosinus
    similarity = np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
    
    # Clipper pour garantir [-1, 1] (erreurs d'arrondi numérique)
    return np.clip(similarity, -1.0, 1.0)


def get_embeddings(client: openai.OpenAI, 
                   texts: list[str], 
                   model: str = EMBEDDING_MODEL,
                   ) -> np.ndarray:
    """ Récupère les embeddings pour une liste de textes via l'API OpenAI. """
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        # Vérifier qu'on a bien reçu des embeddings
        if not response.data:
            raise RuntimeError("Aucun embedding reçu de l'API")
        
        embeddings = [np.array(embedding.embedding) for embedding in response.data]
        return np.array(embeddings)
    except Exception as e:
        raise RuntimeError(f"Échec de récupération des embeddings: {str(e)}")


def simulate_cosine_similarities(
    client: openai.OpenAI,
    text_pairs: list[Tuple[str, str]],
    model: str = EMBEDDING_MODEL,
) -> None:
    """
    Lance une série de comparaisons de similarité cosinus sur des couples de textes.

    Parameters:
    client (openai.OpenAI): Instance du client OpenAI.
    text_pairs (list[Tuple[str, str]]): Couples de textes à comparer.
    model (str): Modèle d'embeddings à utiliser.
    """

    print(f"\n=== SIMULATION AVEC {model} ===")
    for idx, (text_a, text_b) in enumerate(text_pairs, start=1):
        embeddings = get_embeddings(client, [text_a, text_b], model=model)
        
        # Diagnostics détaillés des embeddings
        print(f"DEBUG - Paire {idx}:")
        print(f"  Forme des embeddings: {embeddings.shape}")
        
        similarity = get_cosine_similarity(embeddings[0], embeddings[1])
        print(f"Paire {idx}: similarité = {similarity:.4f}")
        print(f"   ↳ « {text_a} »")
        print(f"   ↳ « {text_b} »\n")

# %%
if __name__ == "__main__":
    # Instanciation directe du client OpenAI, pointant vers le serveur local.
    client = openai.OpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        timeout=100,
    )

    text_pairs = [
        # === COUPLES SÉMANTIQUEMENT TRÈS PROCHES ===
        ("Le chat dort sur le canapé", "Un félin sommeille sur le sofa"),
        ("Paris est la capitale de la France", "La capitale française est Paris"),
        ("La voiture roule vite", "Le véhicule se déplace à grande vitesse"),
        ("Il pleut dehors", "La pluie tombe à l'extérieur"),
        
        # === COUPLES SÉMANTIQUEMENT ÉLOIGNÉS ===
        ("Je prépare un gâteau au chocolat", "Les éléphants vivent en Afrique"),
        ("Je déteste les embouteillages", "La photosynthèse produit de l'oxygène"),
        ("Le soleil brille intensément", "Le cours du bitcoin est en hausse"),
        ("J'aime les mathématiques", "La lune est un satellite"),
        
        # === COUPLES NEUTRES/AMBIGUS ===
        ("Le livre est sur la table", "La table est dans la cuisine"),
        ("Je mange une pomme", "Je bois un café"),

        # === COUPLE SÉMANTIQUEMENT OPPOSÉ ===
        ("Ce film est excellent", "Ce film est horrible"),
    ]

    # Simulation avec deux modèles d'embeddings pour comparaison
    for model in (
        # "text-embedding-nomic-embed-text-v1.5",
        # "text-embedding-nomic-embed-text-v2-moe",
        "charlesli_-_openelm-1_1b-dpo-full-max-reward-most-similar",
    ):
        simulate_cosine_similarities(client, text_pairs, model=model)

    print("\n=== SIMULATION TERMINÉE ===")

# %% 
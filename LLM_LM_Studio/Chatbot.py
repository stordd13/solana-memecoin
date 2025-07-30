# Author: Jeff

# %%
import openai
from dataclasses import dataclass


@dataclass
class ChatConfig:
    """Configuration pour le chat completion."""
    model: str = "gemma-3n-e2b-it-text"
    temperature: float = 0.3
    max_tokens: int = 500
    verbose: bool = True


@dataclass
class Specialist:
    """Définit un profil de spécialiste avec son prompt système."""
    name: str
    role: str
    system_prompt: str
    temperature: float = 0.3  # Peut être ajusté par spécialiste


def stream_chat_completion(
    client: openai.OpenAI,
    config: ChatConfig,
    messages: list[dict[str, str]] = None,
    system_message: str = None,
    user_prompt: str = None,
) -> str:
    """
    Exécute un chat complet en streaming avec l'API OpenAI et retourne le contenu généré.
    
    Parameters:
    client (openai.OpenAI): Client OpenAI configuré.
    config (ChatConfig): Objet de configuration pour le modèle.
    messages (list[dict]): Historique de messages au format rôle/contenu.
    system_message (str): Prompt système pour initialiser la conversation.
    user_prompt (str): Prompt utilisateur obligatoire si messages non fourni.
    verbose (bool): Affiche le résultat en temps réel si True.
    
    Returns:
    str: Contenu généré concaténé.
    
    Exceptions:
    ValueError: Si ni messages ni user_prompt ne sont fournis.
    RuntimeError: En cas d'erreur de l'API.
    """
    try:
        if not messages:
            if not user_prompt:
                raise ValueError("Au moins un prompt utilisateur ou des messages doivent être fournis")
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_prompt})

        stream = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True
        )

        full_response = []
        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                full_response.append(content)
                if config.verbose:
                    print(content, end="", flush=True)

        return ''.join(full_response)

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération du chat: {str(e)}")

# Ajout : boucle interactive pour échanges multiples avec le modèle

# Définition des spécialistes pré-configurés
SPECIALISTS = {
    "lead_data": Specialist(
        name="Lead Data Scientist",
        role="lead_data",
        system_prompt=(
            "Tu es un Lead Data Scientist 10 ans d'expérience expert en trading "
            "de maché financier et de cryptomonnaies, avec de grosse volatilité."
            "Tu maîtrises parfaitement Python, SQL, et les frameworks ML/DL/RL "
            "(scikit-learn, TensorFlow, PyTorch). Tu es spécialisé dans la conception "
            "de modèles de deep learning/ reinforcement learning "
        ),
        temperature=0.3
    ),
    "tech_lead": Specialist(
        name="Tech Lead",
        role="tech_lead",
        system_prompt=(
            "Tu es un Tech Lead senior avec expertise en architecture logicielle. "
            "Tu maîtrises les design patterns, les principes SOLID, et les meilleures "
            "pratiques de développement. Tu es expert en Python, Java, et technologies "
            "cloud (AWS, GCP, Azure). Tu guides sur les choix techniques, la scalabilité, "
            "et la qualité du code."
        ),
        temperature=0.2
    ),
    "dev": Specialist(
        name="Développeur Full-Stack",
        role="dev",
        system_prompt=(
            "Tu es un développeur full-stack expérimenté. Tu maîtrises Python, "
            "JavaScript/TypeScript, les frameworks web (Django, FastAPI, React, Vue.js), "
            "et les bases de données. Tu écris du code propre, testé et documenté. "
            "Tu aides à résoudre des bugs et à implémenter des fonctionnalités."
        ),
        temperature=0.4
    )
}


def interactive_chat(
    client: openai.OpenAI,
    config: ChatConfig,
    specialists: dict[str, Specialist] = None,
) -> None:
    """
    Lance une session interactive de chat similaire à ChatGPT.

    Parameters:
        client (openai.OpenAI): Client configuré pour l'API OpenAI.
        config (ChatConfig): Paramètres du modèle.
        specialists (dict[str, Specialist]): Dictionnaire des spécialistes disponibles.

    Exceptions:
        RuntimeError: Propagée depuis stream_chat_completion.
    """
    if specialists is None:
        specialists = SPECIALISTS
    
    # Sélection du spécialiste initial
    print("\n--- Spécialistes disponibles ---")
    for key, spec in specialists.items():
        print(f"  [{key}] - {spec.name}")
    
    current_specialist = None
    while not current_specialist:
        choice = input("\nChoisissez un spécialiste: ").strip().lower()
        if choice in specialists:
            current_specialist = specialists[choice]
            config.temperature = current_specialist.temperature
            print(f"\n✓ {current_specialist.name} activé!\n")
        else:
            print("Choix invalide. Veuillez réessayer.")
    
    messages: list[dict[str, str]] = [{"role": "system", "content": current_specialist.system_prompt}]

    while True:
        try:
            user_input = input(f"\nUtilisateur ({current_specialist.role}) >> ")
            if user_input.lower() in {"q", ""}:
                print("Fin de la session.")
                break
            
            # Commande pour changer de spécialiste
            if user_input.lower().startswith("/switch"):
                print("\n--- Spécialistes disponibles ---")
                for key, spec in specialists.items():
                    print(f"  [{key}] - {spec.name}")
                
                choice = input("\nChoisissez un nouveau spécialiste: ").strip().lower()
                if choice in specialists:
                    current_specialist = specialists[choice]
                    config.temperature = current_specialist.temperature
                    messages = [{"role": "system", "content": current_specialist.system_prompt}]
                    print(f"\n✓ Changé pour {current_specialist.name}!\n")
                else:
                    print("Choix invalide. Spécialiste non changé.")
                continue

            messages.append({"role": "user", "content": user_input})

            print("\n" + "--- Réponse du modèle ---".center(50, "-"))
            assistant_response = stream_chat_completion(
                client=client,
                config=config,
                messages=messages,
            )
            print("\n" + "--- Fin de la réponse ---".center(50, "-"))

            messages.append({"role": "assistant", "content": assistant_response})
        except KeyboardInterrupt:
            print("\nSession interrompue par l'utilisateur.")
            break
        except RuntimeError as exc:
            print(f"\nUne erreur d'exécution est survenue : {exc}")
        except Exception as exc:
            print(f"\nUne erreur inattendue est survenue : {exc}")

# %%
if __name__ == "__main__":
    try:
        client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="")

        chat_config = ChatConfig(
            model="gemma-3n-e2b-it-text",
            temperature=0.3,
            max_tokens=400,
            verbose=True,
        )

        # Affichage des commandes disponibles
        print("\n--- Commandes disponibles ---")
        print("  /switch - Changer de spécialiste")
        print("  q ou Entrée - Quitter")
        print("\n")
        
        interactive_chat(client=client, config=chat_config, specialists=SPECIALISTS)
    except Exception as exc:
        print(f"Une erreur d'initialisation est survenue : {exc}")

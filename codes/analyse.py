"""
analyse.py
─────────────────────────────────────────────────────────────────────────────
Génération et analyse structurelle des poèmes produits par le modèle.

Ce fichier couvre :
    1. Chargement du modèle entraîné
    2. Génération de poèmes (température, top-k)
    3. Analyse de la longueur des vers
    4. Analyse de la structure des strophes
    5. Analyse des schémas de rimes
    6. Génération d'exemples avec prompts

L'objectif n'est pas simplement d'admirer les poèmes générés, mais de vérifier
de façon systématique si les propriétés structurelles du corpus d'entraînement
réapparaissent dans les textes produits par le modèle.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from model import GPT, dispositif


# ─────────────────────────────────────────────
# 1. Chargement du modèle entraîné
# ─────────────────────────────────────────────

checkpoint   = torch.load('modele_gpt_poesie.pt', map_location=dispositif)
config       = checkpoint['config']
car_vers_idx = checkpoint['vocabulaire']
idx_vers_car = {i: c for c, i in car_vers_idx.items()}

encoder = lambda texte   : [car_vers_idx[c] for c in texte if c in car_vers_idx]
decoder = lambda indices : ''.join([idx_vers_car[i] for i in indices])

modele = GPT(**config).to(dispositif)
modele.load_state_dict(checkpoint['poids'])
modele.eval()
print("Modèle chargé.")
print(f"Vocabulaire : {len(car_vers_idx)} caractères")


# ─────────────────────────────────────────────
# 2. Génération de poèmes
# ─────────────────────────────────────────────
# La génération suit la procédure autorégressive standard : à chaque pas,
# le modèle produit une distribution sur le vocabulaire, on échantillonne
# un caractère selon cette distribution et on l'ajoute à la séquence.
# ─────────────────────────────────────────────

def generer_poeme(prompt_texte='', longueur=500, temperature=0.8, top_k=40):
    """
    Génère un poème à partir d'un prompt textuel.

    Paramètres
    ----------
    prompt_texte : texte de départ (vide = génération libre)
    longueur     : nombre de caractères à générer
    temperature  : créativité de la génération
    top_k        : nombre de candidats conservés à chaque pas
    """
    if prompt_texte:
        indices_prompt = torch.tensor(
            encoder(prompt_texte), dtype=torch.long, device=dispositif
        ).unsqueeze(0)
    else:
        indices_prompt = torch.zeros((1, 1), dtype=torch.long, device=dispositif)

    with torch.no_grad():
        indices_generes = modele.generer(
            indices_prompt, longueur, temperature=temperature, top_k=top_k
        )
    return decoder(indices_generes[0].tolist())

print("\n=== Impact de la température ===")
for temperature in [0.5, 0.8, 1.0, 1.2]:
    print(f"\n{'='*50}")
    print(f"Température : {temperature}")
    print('='*50)
    print(generer_poeme(longueur=300, temperature=temperature))


# ─────────────────────────────────────────────
# 3. Analyse de la longueur des vers
# ─────────────────────────────────────────────

def extraire_vers(texte):
    """Extrait les vers non vides d'un texte poétique."""
    return [ligne for ligne in texte.split('\n') if ligne.strip()]

def analyser_longueur_vers(texte):
    """Retourne la liste des longueurs de vers d'un texte."""
    return [len(vers) for vers in extraire_vers(texte)]

# Génération d'un large échantillon pour l'analyse
texte_genere     = generer_poeme(longueur=3000, temperature=0.8)
echantillon_corpus = decoder(
    torch.load('modele_gpt_poesie.pt', map_location=dispositif)
    .get('historique', {})
    .get('steps', [])
) if False else generer_poeme(longueur=3000, temperature=0.8)

# Rechargement propre d'un échantillon du corpus depuis le checkpoint
import entrainement as ent
echantillon_corpus = decoder(ent.donnees_entrainement[:3000].tolist())

longueurs_corpus  = analyser_longueur_vers(echantillon_corpus)
longueurs_generes = analyser_longueur_vers(texte_genere)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(longueurs_corpus,  bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Longueur des vers — Corpus')
axes[0].set_xlabel('Nombre de caractères')
axes[0].set_ylabel('Fréquence')
axes[0].grid(True)

axes[1].hist(longueurs_generes, bins=30, color='coral', edgecolor='white')
axes[1].set_title('Longueur des vers — Texte généré')
axes[1].set_xlabel('Nombre de caractères')
axes[1].set_ylabel('Fréquence')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('figures/longueur_vers.png', dpi=150)
plt.show()

print(f"\nLongueur moyenne — Corpus  : {sum(longueurs_corpus)/len(longueurs_corpus):.1f} caractères")
print(f"Longueur moyenne — Générés : {sum(longueurs_generes)/len(longueurs_generes):.1f} caractères")


# ─────────────────────────────────────────────
# 4. Analyse de la structure des strophes
# ─────────────────────────────────────────────
# Une strophe est délimitée par une ligne vide. On compare le nombre de vers
# par strophe dans le corpus et dans les textes générés.
# ─────────────────────────────────────────────

def extraire_strophes(texte):
    """Extrait les strophes délimitées par des lignes vides."""
    strophes = texte.strip().split('\n\n')
    return [s.strip() for s in strophes if s.strip()]

def analyser_strophes(texte):
    """Retourne la liste du nombre de vers par strophe."""
    return [len(extraire_vers(s)) for s in extraire_strophes(texte)]

tailles_corpus  = analyser_strophes(echantillon_corpus)
tailles_generes = analyser_strophes(texte_genere)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(tailles_corpus,  bins=range(1, 15), color='steelblue',
             edgecolor='white', align='left')
axes[0].set_title('Vers par strophe — Corpus')
axes[0].set_xlabel('Nombre de vers')
axes[0].set_ylabel('Fréquence')
axes[0].grid(True)

axes[1].hist(tailles_generes, bins=range(1, 15), color='coral',
             edgecolor='white', align='left')
axes[1].set_title('Vers par strophe — Texte généré')
axes[1].set_xlabel('Nombre de vers')
axes[1].set_ylabel('Fréquence')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('figures/structure_strophes.png', dpi=150)
plt.show()

print(f"\nTaille moyenne des strophes — Corpus  : {sum(tailles_corpus)/len(tailles_corpus):.1f} vers")
print(f"Taille moyenne des strophes — Générés : {sum(tailles_generes)/len(tailles_generes):.1f} vers")


# ─────────────────────────────────────────────
# 5. Analyse des schémas de rimes
# ─────────────────────────────────────────────
# On approxime les rimes en comparant les derniers caractères de chaque vers
# (après suppression de la ponctuation finale). Si deux vers partagent la
# même terminaison, ils riment. On encode le schéma en lettres : ABAB, AABB...
# ─────────────────────────────────────────────

def extraire_fin_vers(vers, n_chars=3):
    """Extrait les n derniers caractères d'un vers (approximation de la rime)."""
    vers = vers.strip().rstrip('.,;:!?')
    return vers[-n_chars:].lower() if len(vers) >= n_chars else vers.lower()

def detecter_schema_rimes(strophe):
    """
    Détecte le schéma de rimes d'une strophe.
    Retourne une chaîne de lettres (ex: 'ABAB', 'AABB').
    """
    vers = extraire_vers(strophe)
    if len(vers) < 2:
        return None

    fins            = [extraire_fin_vers(v) for v in vers]
    schema          = []
    lettre_map      = {}
    lettre_courante = ord('A')

    for fin in fins:
        if fin not in lettre_map:
            lettre_map[fin] = chr(lettre_courante)
            lettre_courante += 1
        schema.append(lettre_map[fin])

    return ''.join(schema)

def analyser_rimes(texte):
    strophes = extraire_strophes(texte)
    schemas  = [detecter_schema_rimes(s) for s in strophes]
    return [s for s in schemas if s]

schemas_corpus  = analyser_rimes(echantillon_corpus)
schemas_generes = analyser_rimes(texte_genere)

top_corpus  = Counter(schemas_corpus).most_common(8)
top_generes = Counter(schemas_generes).most_common(8)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar([s for s, _ in top_corpus],  [c for _, c in top_corpus],
            color='steelblue')
axes[0].set_title('Schémas de rimes — Corpus')
axes[0].set_xlabel('Schéma')
axes[0].set_ylabel('Fréquence')
axes[0].grid(True, axis='y')

axes[1].bar([s for s, _ in top_generes], [c for _, c in top_generes],
            color='coral')
axes[1].set_title('Schémas de rimes — Texte généré')
axes[1].set_xlabel('Schéma')
axes[1].set_ylabel('Fréquence')
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('figures/schemas_rimes.png', dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 6. Exemples de poèmes générés avec prompts
# ─────────────────────────────────────────────
PROMPTS = [
    "Demain, dès l'aube",
    "Les sanglots longs",
    "Mon enfant, ma sœur",
    "Je suis la plaie et le couteau",
]

print("\n=== Poèmes générés avec prompts ===")
for prompt in PROMPTS:
    print(f"\n{'='*50}")
    print(f"Prompt : « {prompt} »")
    print('='*50)
    print(generer_poeme(
        prompt_texte=prompt,
        longueur=300,
        temperature=0.8,
        top_k=40,
    ))
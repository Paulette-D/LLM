"""
entrainement.py
─────────────────────────────────────────────────────────────────────────────
Constitution du corpus, entraînement et exploration des hyperparamètres.

Ce fichier couvre :
    1. Téléchargement et nettoyage du corpus poétique (Projet Gutenberg)
    2. Tokenisation au niveau des caractères
    3. Découpage train / validation et chargeur de batches
    4. Boucle d'entraînement principale avec suivi des métriques
    5. Exploration de l'impact des hyperparamètres
    6. Visualisation des courbes d'apprentissage
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import requests
import re
import math
import matplotlib.pyplot as plt
from model import GPT, dispositif


# ─────────────────────────────────────────────
# 1. Téléchargement du corpus
# ─────────────────────────────────────────────
# Textes libres de droits récupérés depuis le Projet Gutenberg.
# Objectif : 5-10 Mo pour un entraînement solide.
# ─────────────────────────────────────────────

SOURCES = {
    'Hugo - Les Contemplations'      : 'https://www.gutenberg.org/cache/epub/8050/pg8050.txt',
    'Baudelaire - Les Fleurs du Mal' : 'https://www.gutenberg.org/cache/epub/6099/pg6099.txt',
    'Verlaine - Poèmes Saturniens'   : 'https://www.gutenberg.org/cache/epub/5264/pg5264.txt',
    'Verlaine - Fêtes Galantes'      : 'https://www.gutenberg.org/cache/epub/5267/pg5267.txt',
    'Rimbaud - Poésies'              : 'https://www.gutenberg.org/cache/epub/6828/pg6828.txt',
}

def telecharger_texte(url):
    reponse          = requests.get(url, timeout=10)
    reponse.encoding = 'utf-8'
    return reponse.text

corpus_brut = []
for titre, url in SOURCES.items():
    try:
        texte = telecharger_texte(url)
        corpus_brut.append(texte)
        print(f"✓ {titre} — {len(texte):,} caractères")
    except Exception as e:
        print(f"✗ {titre} — erreur : {e}")


# ─────────────────────────────────────────────
# 2. Nettoyage du corpus
# ─────────────────────────────────────────────
# Suppression des en-têtes Gutenberg, normalisation des fins de ligne.
# On préserve les sauts de ligne et séparations de strophes — essentiels
# pour étudier la structure poétique.
# ─────────────────────────────────────────────

def nettoyer_gutenberg(texte):
    debut = re.search(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG', texte)
    fin   = re.search(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG', texte)
    if debut:
        texte = texte[debut.end():]
    if fin:
        texte = texte[:fin.start()]
    texte = texte.replace('\r\n', '\n').replace('\r', '\n')
    texte = re.sub(r'\n{4,}', '\n\n\n', texte)
    return texte.strip()

corpus_nettoye = '\n\n'.join([nettoyer_gutenberg(t) for t in corpus_brut])
print(f"\nTaille après nettoyage : {len(corpus_nettoye) / 1e6:.2f} Mo")
print(f"\nExtrait :\n{corpus_nettoye[:300]}")


# ─────────────────────────────────────────────
# 3. Tokenisation au niveau des caractères
# ─────────────────────────────────────────────

caracteres   = sorted(set(corpus_nettoye))
taille_vocab = len(caracteres)
print(f"\nTaille du vocabulaire : {taille_vocab} caractères")
print(f"Caractères : {''.join(caracteres)}")

car_vers_idx = {c: i for i, c in enumerate(caracteres)}
idx_vers_car = {i: c for i, c in enumerate(caracteres)}

encoder = lambda texte   : [car_vers_idx[c] for c in texte]
decoder = lambda indices : ''.join([idx_vers_car[i] for i in indices])

donnees = torch.tensor(encoder(corpus_nettoye), dtype=torch.long)
print(f"Nombre total de tokens : {len(donnees):,}")


# ─────────────────────────────────────────────
# 4. Découpage train / validation
# ─────────────────────────────────────────────

n                    = int(0.9 * len(donnees))
donnees_entrainement = donnees[:n]
donnees_validation   = donnees[n:]

print(f"Entraînement : {len(donnees_entrainement):,} tokens")
print(f"Validation   : {len(donnees_validation):,} tokens")


# ─────────────────────────────────────────────
# 5. Chargeur de batches
# ─────────────────────────────────────────────
def obtenir_batch(split, taille_batch=32, longueur_contexte=256):
    donnees_split = (
        donnees_entrainement if split == 'entrainement'
        else donnees_validation
    )
    indices = torch.randint(len(donnees_split) - longueur_contexte, (taille_batch,))
    x = torch.stack([donnees_split[i:i+longueur_contexte]     for i in indices])
    y = torch.stack([donnees_split[i+1:i+longueur_contexte+1] for i in indices])
    return x.to(dispositif), y.to(dispositif)

# Vérif
x, y = obtenir_batch('entrainement')
print(f"\nForme de x : {x.shape}")
print(f"Forme de y : {y.shape}")


# ─────────────────────────────────────────────
# 6. Configuration et instanciation du modèle
# ─────────────────────────────────────────────

CONFIG = {
    'taille_vocab'      : taille_vocab,
    'd_modele'          : 256,
    'n_tetes'           : 8,
    'n_couches'         : 6,
    'longueur_contexte' : 256,
    'dropout'           : 0.1,
    'mode'              : 'optimisee',
}

TAILLE_BATCH       = 32
N_STEPS            = 50000
TAUX_APPRENTISSAGE = 3e-4
INTERVALLE_EVAL    = 500
INTERVALLE_GENE    = 2000
N_STEPS_EVAL       = 50

modele     = GPT(**CONFIG).to(dispositif)
optimiseur = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE)

n_params = sum(p.numel() for p in modele.parameters())
print(f"Nombre de paramètres : {n_params:,}")

def obtenir_taux(step, n_warmup=1000):
    if step < n_warmup:
        return step / n_warmup
    return 1.0

planificateur = torch.optim.lr_scheduler.LambdaLR(optimiseur, obtenir_taux)


# ─────────────────────────────────────────────
# 7. Fonctions utilitaires
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluer():
    """Calcule la perte moyenne sur le corpus de validation."""
    modele.eval()
    pertes = []
    for _ in range(N_STEPS_EVAL):
        x, y = obtenir_batch('validation', TAILLE_BATCH, CONFIG['longueur_contexte'])
        _, perte = modele(x, y)
        pertes.append(perte.item())
    modele.train()
    return sum(pertes) / len(pertes)

def generer_exemple(longueur=200, temperature=0.8):
    """Génère un extrait de poème à partir d'un prompt vide."""
    modele.eval()
    prompt  = torch.zeros((1, 1), dtype=torch.long, device=dispositif)
    indices = modele.generer(prompt, longueur, temperature=temperature)[0].tolist()
    modele.train()
    return decoder(indices)


# ─────────────────────────────────────────────
# 8. Boucle d'entraînement principale
# ─────────────────────────────────────────────
# On suit à la fois la perte (cross-entropie) et le BPC (bits par caractère),
# défini comme perte / log(2). Le BPC est plus interprétable : il mesure
# combien de bits le modèle utilise en moyenne pour prédire chaque caractère.
# ─────────────────────────────────────────────

historique = {
    'steps'       : [],
    'perte_train' : [],
    'perte_val'   : [],
    'bpc_train'   : [],
    'bpc_val'     : [],
}

modele.train()
for step in range(N_STEPS):
    x, y = obtenir_batch('entrainement', TAILLE_BATCH, CONFIG['longueur_contexte'])

    _, perte = modele(x, y)
    optimiseur.zero_grad()
    perte.backward()

    # Écrêtage du gradient : évite les explosions de gradient
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)

    optimiseur.step()
    planificateur.step()

    if step % INTERVALLE_EVAL == 0:
        perte_val   = evaluer()
        perte_train = perte.item()
        bpc_train   = perte_train / math.log(2)
        bpc_val     = perte_val   / math.log(2)

        historique['steps'].append(step)
        historique['perte_train'].append(perte_train)
        historique['perte_val'].append(perte_val)
        historique['bpc_train'].append(bpc_train)
        historique['bpc_val'].append(bpc_val)

        print(
            f"Step {step:>6} | "
            f"Perte train : {perte_train:.4f} | "
            f"Perte val : {perte_val:.4f} | "
            f"BPC val : {bpc_val:.4f} | "
            f"LR : {planificateur.get_last_lr()[0] * TAUX_APPRENTISSAGE:.2e}"
        )

    if step % INTERVALLE_GENE == 0:
        print(f"\n--- Exemple généré (step {step}) ---")
        print(generer_exemple())
        print("------------------------------------\n")


# ─────────────────────────────────────────────
# 9. Visualisation des courbes d'apprentissage
# ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(historique['steps'], historique['perte_train'], label='Entraînement')
ax1.plot(historique['steps'], historique['perte_val'],   label='Validation')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Perte (cross-entropie)')
ax1.set_title('Évolution de la perte')
ax1.legend()
ax1.grid(True)

ax2.plot(historique['steps'], historique['bpc_train'], label='Entraînement')
ax2.plot(historique['steps'], historique['bpc_val'],   label='Validation')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Bits par caractère (BPC)')
ax2.set_title('Évolution du BPC')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('figures/courbes_apprentissage.png', dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 10. Sauvegarde du modèle
# ─────────────────────────────────────────────

torch.save({
    'config'      : CONFIG,
    'poids'       : modele.state_dict(),
    'vocabulaire' : car_vers_idx,
    'historique'  : historique,
}, 'modele_gpt_poesie.pt')
print("Modèle sauvegardé.")


# ─────────────────────────────────────────────
# 11. Exploration des hyperparamètres
# ─────────────────────────────────────────────
# On étudie l'impact de trois hyperparamètres : nombre de couches,
# dimension cachée et taux de dropout. 
# ─────────────────────────────────────────────

def entrainer_configuration(config, n_steps=10000, taille_batch=32):
    modele_exp    = GPT(**config).to(dispositif)
    optimiseur_exp = torch.optim.AdamW(modele_exp.parameters(), lr=3e-4)
    historique_exp = {'steps': [], 'perte_val': [], 'bpc_val': []}

    modele_exp.train()
    for step in range(n_steps):
        x, y = obtenir_batch('entrainement', taille_batch, config['longueur_contexte'])
        _, perte = modele_exp(x, y)
        optimiseur_exp.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(modele_exp.parameters(), 1.0)
        optimiseur_exp.step()

        if step % 500 == 0:
            modele_exp.eval()
            pertes_val = []
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = obtenir_batch('validation', taille_batch, config['longueur_contexte'])
                    _, pv  = modele_exp(xv, yv)
                    pertes_val.append(pv.item())
            modele_exp.train()
            perte_val = sum(pertes_val) / len(pertes_val)
            historique_exp['steps'].append(step)
            historique_exp['perte_val'].append(perte_val)
            historique_exp['bpc_val'].append(perte_val / math.log(2))

    n_params = sum(p.numel() for p in modele_exp.parameters())
    print(
        f"Config : {config['n_couches']} couches | "
        f"d={config['d_modele']} | "
        f"dropout={config['dropout']} | "
        f"Params : {n_params:,} | "
        f"BPC val final : {historique_exp['bpc_val'][-1]:.4f}"
    )
    return historique_exp

CONFIG_BASE = {**CONFIG}

# Expérience 1 — nombre de couches
print("\n=== Expérience 1 : nombre de couches ===")
resultats_couches = {}
for n_couches in [2, 4, 6, 8]:
    config_exp = {**CONFIG_BASE, 'n_couches': n_couches}
    resultats_couches[n_couches] = entrainer_configuration(config_exp)

# Expérience 2 — dimension cachée
print("\n=== Expérience 2 : dimension cachée ===")
resultats_dim = {}
for d_modele in [128, 256, 512]:
    n_tetes    = 8 if d_modele >= 256 else 4
    config_exp = {**CONFIG_BASE, 'd_modele': d_modele, 'n_tetes': n_tetes}
    resultats_dim[d_modele] = entrainer_configuration(config_exp)

# Expérience 3 — taux de dropout
print("\n=== Expérience 3 : taux de dropout ===")
resultats_dropout = {}
for dropout in [0.0, 0.1, 0.2, 0.3]:
    config_exp = {**CONFIG_BASE, 'dropout': dropout}
    resultats_dropout[dropout] = entrainer_configuration(config_exp)


# ─────────────────────────────────────────────
# 12. Visualisation des hyperparamètres
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for n_couches, hist in resultats_couches.items():
    ax.plot(hist['steps'], hist['bpc_val'], label=f'{n_couches} couches')
ax.set_title('Impact du nombre de couches')
ax.set_xlabel('Steps')
ax.set_ylabel('BPC validation')
ax.legend()
ax.grid(True)

ax = axes[1]
for d_modele, hist in resultats_dim.items():
    ax.plot(hist['steps'], hist['bpc_val'], label=f'd={d_modele}')
ax.set_title('Impact de la dimension cachée')
ax.set_xlabel('Steps')
ax.set_ylabel('BPC validation')
ax.legend()
ax.grid(True)

ax = axes[2]
for dropout, hist in resultats_dropout.items():
    ax.plot(hist['steps'], hist['bpc_val'], label=f'dropout={dropout}')
ax.set_title('Impact du dropout')
ax.set_xlabel('Steps')
ax.set_ylabel('BPC validation')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('figures/hyperparametres.png', dpi=150)
plt.show()
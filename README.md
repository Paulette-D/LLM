# Lab 3 — Modèle GPT au niveau des caractères
**Where Hidden Rhythms in Silent Vectors Lie**

Université Paris Cité — M1 Logos — 2026

## Description
Implémentation et analyse d'un modèle de langue autorégressif (GPT) au niveau
des caractères, entraîné sur un corpus de poésie française (Hugo, Baudelaire,
Verlaine, Rimbaud). L'objectif est d'étudier l'émergence de structures
linguistiques au cours de l'apprentissage.

## Structure du dépôt
Le code a été divisé en trois fichiers distincts, disponibles dans le dossier `codes/` :
- `codes/modele.py` — architecture du modèle (embeddings, attention, FFN, GPT)
- `codes/entrainement.py` — corpus, boucle d'entraînement, hyperparamètres
- `codes/analyse.py` — génération et analyse structurelle des poèmes

Les autres éléments du dépôt :
- `figures/` — visualisations produites au cours des expériences
- `lab3_gpt_poesie.ipynb` — notebook Google Colab utilisé pour l'exécution

## Exécution
L'entraînement a été réalisé sur Google Colab avec un GPU T4.
Ouvrir `lab3_gpt_poesie.ipynb` directement dans Google Colab.

## Auteur
Paulette Diouf

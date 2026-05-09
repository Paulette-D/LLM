"""
modele.py
─────────────────────────────────────────────────────────────────────────────
Architecture du modèle GPT au niveau des caractères.

Ce fichier implémente les composants d'un Transformer décodeur (decoder-only),
architecture sur laquelle reposent les modèles de la famille GPT. Le modèle
opère au niveau des caractères : chaque token est un caractère unique, ce qui
permet de conserver un vocabulaire réduit et d'éviter toute tokenisation complexe.

Composants implémentés :
    1. Embeddings          — représentation des tokens et des positions
    2. Attention causale   — mécanisme d'attention masqué multi-têtes
    3. Bloc Feed-Forward   — transformation non-linéaire position par position
    4. Bloc Transformer    — assemblage avec connexions résiduelles et LayerNorm
    5. Modèle GPT          — empilement complet + tête de prédiction
    6. Benchmark           — comparaison des deux implémentations de l'attention
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


dispositif = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# 1. Embeddings
# ─────────────────────────────────────────────
# Un réseau de neurones ne peut pas traiter directement des indices entiers :
# il lui faut des vecteurs continus. La couche d'embeddings assure cette
# transformation en combinant deux matrices apprises :
#   - emb_token    : associe un vecteur à chaque caractère du vocabulaire
#   - emb_position : associe un vecteur à chaque position dans la séquence
#
# L'attention étant invariante à l'ordre, l'embedding positionnel est
# indispensable pour que le modèle sache où se trouve chaque token.
# Nous optons pour des embeddings positionnels appris (comme dans GPT-1/2),
# plus simples à implémenter que les variantes sinusoïdales ou RoPE.
#
# Représentation finale : x_t = emb_token(c_t) + emb_position(t)
# ─────────────────────────────────────────────

class Embeddings(nn.Module):
    def __init__(self, taille_vocab, d_modele, longueur_contexte, dropout=0.1):
        super().__init__()
        self.emb_token         = nn.Embedding(taille_vocab, d_modele)
        self.emb_position      = nn.Embedding(longueur_contexte, d_modele)
        self.dropout           = nn.Dropout(dropout)
        self.longueur_contexte = longueur_contexte

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        return self.dropout(self.emb_token(x) + self.emb_position(positions))
        # return : (B, T, d_modele)


# ─────────────────────────────────────────────
# 2. Attention causale multi-têtes
# ─────────────────────────────────────────────
# L'attention permet à chaque position d'agréger de l'information provenant
# des autres positions, via des produits scalaires entre requêtes et clés :
#
#   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
#
# La causalité est assurée en masquant les positions futures (scores = -inf
# avant le softmax), ce qui garantit que le token t ne "voit" que les tokens
# 0, ..., t-1 — contrainte indispensable pour la génération autorégressive.
#
# Le mécanisme multi-têtes réalise h calculs d'attention en parallèle dans
# des sous-espaces de dimension d_k = d/h, permettant au modèle de capturer
# différents types de relations (rimes, mètre, syntaxe...).
#
# Deux implémentations sont proposées et comparées :
#   - 'classique'  : masque triangulaire explicite + calcul manuel
#   - 'optimisee'  : F.scaled_dot_product_attention (FlashAttention si dispo)
# ─────────────────────────────────────────────

class AttentionCausaleMultiTetes(nn.Module):
    def __init__(self, d_modele, n_tetes, longueur_contexte, dropout=0.1, mode='optimisee'):
        super().__init__()
        assert d_modele % n_tetes == 0, "d_modele doit être divisible par n_tetes"

        self.n_tetes  = n_tetes
        self.d_tete   = d_modele // n_tetes
        self.d_modele = d_modele
        self.dropout  = dropout
        self.mode     = mode

        # Projection unique pour Q, K, V (plus efficace qu'une projection séparée)
        self.proj_qkv     = nn.Linear(d_modele, 3 * d_modele, bias=False)
        self.proj_sortie  = nn.Linear(d_modele, d_modele, bias=False)
        self.dropout_attn = nn.Dropout(dropout)

        # Masque triangulaire inférieur : position i ne voit que j <= i
        masque = torch.tril(torch.ones(longueur_contexte, longueur_contexte)).bool()
        self.register_buffer('masque_causal', masque)

    def _attention_classique(self, q, k, v):
        # Approche traditionnelle : construction explicite du masque
        T      = q.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_tete)
        scores = scores.masked_fill(~self.masque_causal[:T, :T], float('-inf'))
        poids  = self.dropout_attn(F.softmax(scores, dim=-1))
        return torch.matmul(poids, v)

    def _attention_optimisee(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

    def forward(self, x):
        B, T, _ = x.shape

        q, k, v = self.proj_qkv(x).split(self.d_modele, dim=-1)

        def decouper(t):
            return t.view(B, T, self.n_tetes, self.d_tete).transpose(1, 2)

        q, k, v  = decouper(q), decouper(k), decouper(v)
        contexte = self._attention_classique(q, k, v) if self.mode == 'classique' \
                   else self._attention_optimisee(q, k, v)

        contexte = contexte.transpose(1, 2).contiguous().view(B, T, self.d_modele)
        return self.proj_sortie(contexte)


# ─────────────────────────────────────────────
# 3. Bloc Feed-Forward
# ─────────────────────────────────────────────
# Appliqué indépendamment à chaque position après l'attention.
# Composé de deux projections linéaires avec une activation GELU entre les deux.
# La dimension interne est élargie d'un facteur 4 (standard dans les Transformers),
# ce qui donne au réseau une capacité de représentation supplémentaire.
# ─────────────────────────────────────────────

class BlocFeedForward(nn.Module):
    def __init__(self, d_modele, facteur_expansion=4, dropout=0.1):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(d_modele, facteur_expansion * d_modele),
            nn.GELU(),
            nn.Linear(facteur_expansion * d_modele, d_modele),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.reseau(x)


# ─────────────────────────────────────────────
# 4. Bloc Transformer
# ─────────────────────────────────────────────
# Assemble l'attention et le feed-forward avec :
#   - LayerNorm avant chaque sous-couche (pre-norm, plus stable à l'entraînement)
#   - Connexions résiduelles autour de chaque sous-couche
#
# La connexion résiduelle x = x + f(LayerNorm(x)) permet au gradient de
# circuler directement à travers les couches, facilitant l'apprentissage
# de modèles profonds.
# ─────────────────────────────────────────────

class BlocTransformer(nn.Module):
    def __init__(self, d_modele, n_tetes, longueur_contexte, dropout=0.1, mode='optimisee'):
        super().__init__()
        self.norme_1   = nn.LayerNorm(d_modele)
        self.attention = AttentionCausaleMultiTetes(d_modele, n_tetes, longueur_contexte, dropout, mode)
        self.norme_2   = nn.LayerNorm(d_modele)
        self.ff        = BlocFeedForward(d_modele, dropout=dropout)

    def forward(self, x):
        x = x + self.attention(self.norme_1(x))   # résidu autour de l'attention
        x = x + self.ff(self.norme_2(x))           # résidu autour du feed-forward
        return x


# ─────────────────────────────────────────────
# 5. Modèle GPT complet
# ─────────────────────────────────────────────
# Assemble tous les composants :
#   - Couche d'embeddings
#   - Pile de n_couches blocs Transformer
#   - LayerNorm finale
#   - Tête de prédiction : projection vers le vocabulaire
#
# Le partage de poids entre l'embedding de tokens et la tête de prédiction
# (weight tying) est une technique standard qui réduit le nombre de paramètres
# et améliore généralement les performances.
# ─────────────────────────────────────────────

class GPT(nn.Module):
    def __init__(self, taille_vocab, d_modele, n_tetes, n_couches, longueur_contexte, dropout=0.1, mode='optimisee'):
        super().__init__()
        self.embeddings   = Embeddings(taille_vocab, d_modele, longueur_contexte, dropout)
        self.blocs        = nn.Sequential(*[
            BlocTransformer(d_modele, n_tetes, longueur_contexte, dropout, mode)
            for _ in range(n_couches)
        ])
        self.norme_finale = nn.LayerNorm(d_modele)
        self.tete_lm      = nn.Linear(d_modele, taille_vocab, bias=False)

        self.tete_lm.weight = self.embeddings.emb_token.weight
        self.appliquer_init()

    def appliquer_init(self):
        # Initialisation gaussienne standard pour les Transformers (std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, cibles=None):
        x      = self.embeddings(x)
        x      = self.blocs(x)
        x      = self.norme_finale(x)
        logits = self.tete_lm(x)                                        # (B, T, taille_vocab)
        perte  = None
        if cibles is not None:
            perte = F.cross_entropy(logits.view(-1, logits.size(-1)), cibles.view(-1))
        return logits, perte

    @torch.no_grad()
    def generer(self, x, longueur_max, temperature=1.0, top_k=None):
        """
        Génération autorégressive : à chaque pas, prédit le prochain caractère
        et l'ajoute à la séquence. Deux paramètres contrôlent la génération :
          - temperature : < 1 = plus déterministe, > 1 = plus créatif
          - top_k       : restreint l'échantillonnage aux k meilleurs caractères
        """
        for _ in range(longueur_max):
            x_tronque      = x[:, -self.embeddings.longueur_contexte:]
            logits, _      = self(x_tronque)
            logits         = logits[:, -1, :] / temperature
            if top_k is not None:
                valeurs, _ = torch.topk(logits, top_k)
                logits[logits < valeurs[:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            prochain = torch.multinomial(probs, num_samples=1)
            x        = torch.cat([x, prochain], dim=1)
        return x


# ─────────────────────────────────────────────
# 6. Benchmark des deux implémentations d'attention
# ─────────────────────────────────────────────
# Compare le mode 'classique' (masque triangulaire explicite) et le mode
# 'optimisee' (F.scaled_dot_product_attention / FlashAttention) en termes
# de vitesse d'exécution et de consommation mémoire GPU.
# ─────────────────────────────────────────────

def benchmark_attention(mode, n_repetitions=100):
    B, T, d_modele, n_tetes = 32, 256, 256, 8
    modele = AttentionCausaleMultiTetes(
        d_modele, n_tetes, T, dropout=0.0, mode=mode
    ).to(dispositif)
    entree = torch.randn(B, T, d_modele, device=dispositif)

    with torch.no_grad():
        for _ in range(10):
            modele(entree)

    if dispositif.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    debut = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repetitions):
            modele(entree)
    if dispositif.type == 'cuda':
        torch.cuda.synchronize()
    fin = time.perf_counter()

    temps_ms   = (fin - debut) / n_repetitions * 1000
    memoire_mo = (
        torch.cuda.max_memory_allocated() / 1024**2
        if dispositif.type == 'cuda' else float('nan')
    )
    print(f"[{mode:>10}]  Temps : {temps_ms:.3f} ms  |  Mémoire GPU : {memoire_mo:.1f} Mo")


if __name__ == '__main__':
    print("=== Benchmark attention ===")
    benchmark_attention('classique')
    benchmark_attention('optimisee')
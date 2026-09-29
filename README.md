# TP Probabilités & Statistiques — Régression Logistique

Elias Bonnefoi — ESPCI Paris PSL, 1ère année (2025-2026)

## Contenu

- [`regression_logistique.py`](regression_logistique.py) — code Python (descente de gradient, régression logistique implémentée from scratch)
- `data/` — jeux de données `.mat` (`reglog_data_1.mat`, `reglog_data_2.mat`, `reglog_data_3.mat`), à ajouter localement (non versionnés, voir `.gitignore`)

## Résumé

La régression logistique modélise la probabilité qu'un vecteur `x` appartienne à la classe 1 via un hyperplan séparateur et la fonction sigmoïde :

```
P(c=1|x) = σ(wᵀx + b) = 1 / (1 + exp(-(wᵀx + b)))
```

Les paramètres `(w, b)` sont appris par descente de gradient sur la perte de log-vraisemblance négative (NLL).

Le TP explore trois jeux de données de complexité croissante :

1. **Dataset 1** — linéairement séparable. Étude de l'effet du pas d'apprentissage (`lr`) et du nombre d'itérations sur la convergence.
2. **Dataset 2** — classes qui se chevauchent légèrement. Taux d'erreur résiduel incompressible (~7%) ; mise en évidence de la divergence de la norme des paramètres sur données séparables.
3. **Dataset 3** — classes concentriques (structure non-linéaire). La régression logistique classique échoue (~50% d'erreur) ; un enrichissement polynomial des features (`phi(x) = [x₁, x₂, x₁²+x₂²]`) permet de retrouver une frontière de décision non-linéaire performante.

## Remarque sur l'utilisation de l'IA

L'IA (Gemini) a été utilisée pour débugger certaines cellules de code, et accélérer l'écriture de certaines cellules de code comme la structure des graphiques. Son apport principal a été la mise en forme rapide du code NumPy/Matplotlib et la suggestion du clipping dans la fonction objectif (éviter log(0)). En revanche, l'IA a eu tendance à proposer des implémentations trop génériques (ex. utilisation de sklearn au lieu du code from scratch demandé) et à omettre la conversion des paramètres vers l'espace original pour la visualisation — ce que j'ai dû corriger manuellement.

## Dépendances

```bash
pip install numpy matplotlib scipy
```

## Exécution

```bash
python regression_logistique.py
```

# -*- coding: utf-8 -*-
"""
TP Probabilités & Statistiques — Régression Logistique
Elias Bonnefoi — ESPCI Paris PSL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def plotdata(X, C, w=None, b=None):
    MIN = 0
    MAX = 20
    pos = (C == 1).squeeze()
    neg = (C == 0).squeeze()
    plt.plot(X[0, pos], X[1, pos], "g+")
    plt.plot(X[0, neg], X[1, neg], "r+")
    if w is None or b is None:
        plt.xlim(MIN, MAX)
        plt.ylim(MIN, MAX)
        plt.grid()
        plt.show()
        return
    if w[1] == 0:
        if w[0] == 0:
            print("WARNING cannot plot such line ...")
            return
        x1 = -b / w[0]
        y1 = MAX
        x2 = x1
        y2 = MIN
    else:
        x1 = MAX
        y1 = -(b + w[0] * x1) / w[1]
        x2 = MIN
        y2 = -(b + w[0] * x2) / w[1]
    plt.plot([x1, x2], [y1, y2], "b-")
    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)
    plt.grid()
    plt.show()


def init(std=0.1):
    w = np.random.randn(1, 2) * std
    b = np.random.randn() * std
    return w, b


def init_d(d, std=0.1):
    w = np.random.randn(1, d) * std
    b = np.random.randn() * std
    return w, b


def inference(x, w, b):
    a = w @ x + b
    return 1 / (1 + np.exp(-a))


def objectif(predictions, vraies_classes):
    eps = 1e-12
    p = np.clip(predictions, eps, 1 - eps)
    return -np.sum(vraies_classes * np.log(p) + (1 - vraies_classes) * np.log(1 - p))


def gradient_objectif(vraies_classes, predictions, entrees):
    diff = predictions - vraies_classes
    dw = diff @ entrees.T
    db = np.sum(diff)
    return dw, db


def augment_features(X):
    x1_c = X[0:1, :] - X[0].mean()
    x2_c = X[1:2, :] - X[1].mean()
    r2 = x1_c**2 + x2_c**2
    return np.vstack([x1_c, x2_c, r2])


# -------------------------------------------------------
# Dataset 1 — cas linéairement séparable
# -------------------------------------------------------

dataset = loadmat("data/reglog_data_1.mat")
X = dataset["X"]
C = dataset["C"]

plotdata(X, C)
plotdata(X, C, w=np.ones(2), b=-20)

np.random.seed(42)
w_test, b_test = init()
pred_test = inference(X, w_test, b_test)
print("Shape prédictions:", pred_test.shape)
print("Objectif:", objectif(pred_test, C))
dw_test, db_test = gradient_objectif(C, pred_test, X)
print("dw shape:", dw_test.shape, "db:", db_test)

mu1 = X.mean(axis=1, keepdims=True)
sigma1 = X.std(axis=1, keepdims=True)
X_n = (X - mu1) / sigma1

for n_iter in (100, 1000, 10000):
    np.random.seed(0)
    w, b = init()
    lr = 0.1
    pertes, taux_erreur = [], []

    for i in range(n_iter):
        pred = inference(X_n, w, b)
        pertes.append(objectif(pred, C))
        taux_erreur.append(np.mean((pred >= 0.5).astype(int) != C))
        dw, db = gradient_objectif(C, pred, X_n)
        w = w - lr * dw
        b = b - lr * db

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(pertes)
    axes[0].set_xlabel('Itération')
    axes[0].set_ylabel('NLL')
    axes[0].set_title(f'Fonction de coût – {n_iter} iter')
    axes[0].grid()
    axes[1].plot(taux_erreur)
    axes[1].set_xlabel('Itération')
    axes[1].set_ylabel("Taux d'erreur")
    axes[1].set_title(f"Taux d'erreur – {n_iter} iter")
    axes[1].grid()
    plt.tight_layout()
    plt.show()

    w_orig = w / sigma1.T
    b_orig = float(b) - float(np.dot(w_orig.flatten(), mu1.flatten()))
    plotdata(X, C, w=w_orig.squeeze(), b=b_orig)
    print(f"Perte : {pertes[-1]:.4f} | Erreur : {taux_erreur[-1]*100:.2f}%")


# -------------------------------------------------------
# Dataset 2 — classes qui se chevauchent légèrement
# -------------------------------------------------------

dataset2 = loadmat("data/reglog_data_2.mat")
X2 = dataset2["X"]
C2 = dataset2["C"]

mu2 = X2.mean(axis=1, keepdims=True)
sigma2 = X2.std(axis=1, keepdims=True)
X2_n = (X2 - mu2) / sigma2

plotdata(X2, C2)
# les nuages se chevauchent un peu => pas de séparation parfaite possible

np.random.seed(0)
w, b = init()
lr = 0.01
n_iter = 500
pertes, taux_erreur = [], []

for i in range(n_iter):
    pred = inference(X2_n, w, b)
    pertes.append(objectif(pred, C2))
    pred_classe = (pred >= 0.5).astype(int)
    taux_erreur.append(np.mean(pred_classe != C2))
    dw, db = gradient_objectif(C2, pred, X2_n)
    w = w - lr * dw
    b = b - lr * db

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pertes)
axes[0].set_xlabel('Itération')
axes[0].set_ylabel('NLL')
axes[0].set_title('lr=0.01, 500 itérations')
axes[0].grid()
axes[1].plot(taux_erreur)
axes[1].set_xlabel('Itération')
axes[1].set_ylabel("Taux d'erreur")
axes[1].set_title("Taux d'erreur")
axes[1].grid()
plt.tight_layout()
plt.show()

print(f"Perte finale : {pertes[-1]:.4f}, Taux d'erreur : {taux_erreur[-1]*100:.2f}%")

w_orig = w / sigma2.T
b_orig = float(b) - float(np.dot(w_orig.flatten(), mu2.flatten()))
plotdata(X2, C2, w=w_orig.squeeze(), b=b_orig)

# OBSERVATION (Données 2) :
# - La perte converge correctement grâce à la normalisation
# - Le taux d'erreur se stabilise autour de 7%
# - La frontière de décision linéaire est bien placée
# - Ce taux d'erreur résiduel est normal : les classes se chevauchent
#   légèrement, une droite ne peut pas les séparer parfaitement

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for idx, lr_test in enumerate([0.1, 0.5]):
    np.random.seed(0)
    w, b = init()
    pertes_t = []
    taux_erreur_t = []
    for i in range(500):
        pred = inference(X2_n, w, b)
        pertes_t.append(objectif(pred, C2))
        pred_classe = (pred >= 0.5).astype(int)
        taux_erreur_t.append(np.mean(pred_classe != C2))
        dw, db = gradient_objectif(C2, pred, X2_n)
        w = w - lr_test * dw
        b = b - lr_test * db

    axes[idx, 0].plot(pertes_t)
    axes[idx, 0].set_title(f'Perte - lr={lr_test}')
    axes[idx, 0].set_xlabel('Itération')
    axes[idx, 0].grid()

    axes[idx, 1].plot(taux_erreur_t)
    axes[idx, 1].set_title(f"Taux d'erreur - lr={lr_test}")
    axes[idx, 1].set_xlabel('Itération')
    axes[idx, 1].grid()

    print(f"lr={lr_test} — Perte finale : {pertes_t[-1]:.4f}, Taux d'erreur : {taux_erreur_t[-1]*100:.2f}%")

plt.tight_layout()
plt.show()

np.random.seed(0)
w, b = init()
lr = 0.01
n_iter = 1000
pertes, taux_erreur, normes = [], [], []

for i in range(n_iter):
    pred = inference(X2_n, w, b)
    pertes.append(objectif(pred, C2))
    pred_classe = (pred >= 0.5).astype(int)
    taux_erreur.append(np.mean(pred_classe != C2))
    normes.append(np.sqrt(b**2 + np.sum(w**2)))
    dw, db = gradient_objectif(C2, pred, X2_n)
    w = w - lr * dw
    b = b - lr * db

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(pertes)
axes[0].set_title('Perte (lr=0.01, 1000 iter)')
axes[0].set_xlabel('Itération')
axes[0].grid()
axes[1].plot(taux_erreur)
axes[1].set_title("Taux d'erreur")
axes[1].set_xlabel('Itération')
axes[1].grid()
axes[2].plot(normes)
axes[2].set_title('Norme des paramètres')
axes[2].set_xlabel('Itération')
axes[2].grid()
plt.tight_layout()
plt.show()

print(f"Perte finale : {pertes[-1]:.4f}, Taux d'erreur : {taux_erreur[-1]*100:.2f}%")
print(f"Norme finale des paramètres : {normes[-1]:.4f}")

w_orig = w / sigma2.T
b_orig = float(b) - float(np.dot(w_orig.flatten(), mu2.flatten()))
plotdata(X2, C2, w=w_orig.squeeze(), b=b_orig)

# La norme croît sans se stabiliser : sur des données (presque) linéairement
# séparables, le modèle peut toujours réduire la NLL en augmentant la norme
# de w (frontière géométriquement identique mais probabilités de plus en
# plus proches de 0/1). Il n'existe alors pas de minimum fini pour la NLL.


# -------------------------------------------------------
# Dataset 3 — structure non-linéaire (classes concentriques)
# -------------------------------------------------------

dataset3 = loadmat("data/reglog_data_3.mat")
X3 = dataset3["X"]
C3 = dataset3["C"]
print(X3.shape, C3.shape)

plotdata(X3, C3)
# structure non-linéaire : les classes forment des ensembles concentriques,
# aucune droite ne peut les séparer

mu3 = X3.mean(axis=1, keepdims=True)
sigma3 = X3.std(axis=1, keepdims=True)
X3_n = (X3 - mu3) / sigma3

np.random.seed(0)
w, b = init()
lr = 0.01
n_iter = 1000
pertes, taux_erreur = [], []

for i in range(n_iter):
    pred = inference(X3_n, w, b)
    pertes.append(objectif(pred, C3))
    pred_classe = (pred >= 0.5).astype(int)
    taux_erreur.append(np.mean(pred_classe != C3))
    dw, db = gradient_objectif(C3, pred, X3_n)
    w = w - lr * dw
    b = b - lr * db

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pertes)
axes[0].set_title('Perte - régression logistique classique')
axes[0].set_xlabel('Itération')
axes[0].grid()
axes[1].plot(taux_erreur)
axes[1].set_title("Taux d'erreur")
axes[1].set_xlabel('Itération')
axes[1].grid()
plt.tight_layout()
plt.show()

print(f"Perte finale : {pertes[-1]:.4f}, Taux d'erreur : {taux_erreur[-1]*100:.2f}%")

w_orig = w / sigma3.T
b_orig = float(b) - float(np.dot(w_orig.flatten(), mu3.flatten()))
plotdata(X3, C3, w=w_orig.squeeze(), b=b_orig)

# Le taux d'erreur reste élevé (~17-20%) : une droite ne peut pas séparer
# ces données => il faut enrichir les features avec des termes non-linéaires.

# idée : les classes sont concentriques => la distance au centre est
# discriminante. On construit phi(x) = [x1-cx, x2-cx, (x1-cx)^2+(x2-cx)^2] ;
# dans cet espace augmenté, la frontière circulaire devient un hyperplan.

X3_aug = augment_features(X3)
print("Shape features étendues:", X3_aug.shape)

np.random.seed(0)
w3_aug, b3_aug = init_d(3)
lr = 0.01
n_iter = 10000
pertes, taux_erreur, normes = [], [], []

for i in range(n_iter):
    pred = inference(X3_aug, w3_aug, b3_aug)
    pertes.append(objectif(pred, C3))
    taux_erreur.append(np.mean((pred >= 0.5).astype(int) != C3))
    normes.append(np.sqrt(b3_aug**2 + np.sum(w3_aug**2)))
    dw, db = gradient_objectif(C3, pred, X3_aug)
    w3_aug = w3_aug - lr * dw
    b3_aug = b3_aug - lr * db

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(pertes)
axes[0].set_title('Perte - espace augmenté')
axes[0].set_xlabel('Itération')
axes[0].grid()
axes[1].plot(taux_erreur)
axes[1].set_title("Taux d'erreur")
axes[1].set_xlabel('Itération')
axes[1].grid()
axes[2].plot(normes)
axes[2].set_title('Norme des paramètres')
axes[2].set_xlabel('Itération')
axes[2].grid()
plt.tight_layout()
plt.show()

print(f"Perte finale : {pertes[-1]:.4f} | Erreur : {taux_erreur[-1]*100:.2f}%")

MIN, MAX = 0, 20
x1g, x2g = np.meshgrid(np.linspace(MIN, MAX, 200), np.linspace(MIN, MAX, 200))
X_grid = np.vstack([x1g.ravel(), x2g.ravel()])
X_grid_aug = augment_features(X_grid)
preds_grid = inference(X_grid_aug, w3_aug, b3_aug).reshape(200, 200)

pos = (C3 == 1).squeeze()
neg = (C3 == 0).squeeze()

plt.contourf(x1g, x2g, preds_grid, levels=[0, 0.5, 1], colors=['#ffcccc', '#ccffcc'])
plt.contour(x1g, x2g, preds_grid, levels=[0.5], colors='blue', linewidths=2)
plt.plot(X3[0, pos], X3[1, pos], 'g+', label='Classe 1')
plt.plot(X3[0, neg], X3[1, neg], 'r+', label='Classe 0')
plt.xlim(MIN, MAX)
plt.ylim(MIN, MAX)
plt.legend()
plt.title('Frontière de décision non linéaire – espace augmenté')
plt.grid()
plt.show()

# L'enrichissement de la représentation transforme le problème : dans
# l'espace augmenté, la régression logistique apprend un hyperplan qui
# correspond à une frontière circulaire dans l'espace original. Le taux
# d'erreur tombe sous les 5% contre ~17-20% pour le modèle linéaire.

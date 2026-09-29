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


def phi(X):
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

mu1 = X.mean(axis=1, keepdims=True)
sigma1 = X.std(axis=1, keepdims=True)
X_n = (X - mu1) / sigma1

plotdata(X, C)

np.random.seed(42)
w_test, b_test = init()
pred_test = inference(X, w_test, b_test)
print("Shape prédictions:", pred_test.shape)
print("Objectif:", objectif(pred_test, C))
dw_test, db_test = gradient_objectif(C, pred_test, X)
print("dw shape:", dw_test.shape, "db:", db_test)

np.random.seed(0)
w, b = init()

lr = 0.01
n_iter = 100
pertes = []
taux_erreur = []

for i in range(n_iter):
    pred = inference(X_n, w, b)
    pertes.append(objectif(pred, C))
    pred_classe = (pred >= 0.5).astype(int)
    taux_erreur.append(np.mean(pred_classe != C))
    dw, db = gradient_objectif(C, pred, X_n)
    w = w - lr * dw
    b = b - lr * db

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pertes)
axes[0].set_xlabel('Itération')
axes[0].set_ylabel('NLL')
axes[0].set_title('Fonction de coût')
axes[0].grid()

axes[1].plot(taux_erreur)
axes[1].set_xlabel('Itération')
axes[1].set_ylabel("Taux d'erreur")
axes[1].set_title("Taux d'erreur de classification")
axes[1].grid()
plt.tight_layout()
plt.show()

print(f"Perte finale : {pertes[-1]:.4f}")
print(f"Taux d'erreur final : {taux_erreur[-1]*100:.2f}%")

w_orig = w / sigma1.T
b_orig = float(b) - float(np.dot(w_orig.flatten(), mu1.flatten()))
plotdata(X, C, w=w_orig.squeeze(), b=b_orig)

for lr_test in [0.01, 0.1, 0.5]:
    np.random.seed(0)
    w, b = init()
    pertes_t = []
    taux_t = []
    for i in range(1000):
        pred = inference(X_n, w, b)
        pertes_t.append(objectif(pred, C))
        taux_t.append(np.mean((pred >= 0.5).astype(int) != C))
        dw, db = gradient_objectif(C, pred, X_n)
        w = w - lr_test * dw
        b = b - lr_test * db
    print(f"lr={lr_test} | perte finale={pertes_t[-1]:.4f} | erreur={taux_t[-1]*100:.2f}%")

np.random.seed(0)
w, b = init()
lr = 0.1
n_iter = 1000
pertes = []
taux_erreur = []

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
axes[0].set_title('Perte – lr=0.1, 1000 iter')
axes[0].grid()
axes[1].plot(taux_erreur)
axes[1].set_xlabel('Itération')
axes[1].set_ylabel("Taux d'erreur")
axes[1].set_title('Erreur – lr=0.1, 1000 iter')
axes[1].grid()
plt.tight_layout()
plt.show()

w_orig = w / sigma1.T
b_orig = float(b) - float(np.dot(w_orig.flatten(), mu1.flatten()))
plotdata(X, C, w=w_orig.squeeze(), b=b_orig)
print(f"Perte finale : {pertes[-1]:.4f}, erreur : {taux_erreur[-1]*100:.2f}%")


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

np.random.seed(0)
w, b = init()

lr = 0.01
n_iter = 500
pertes = []
taux_erreur = []

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

"""
OBSERVATION (Données 2) :
- La perte converge correctement grâce à la normalisation
- Le taux d'erreur se stabilise autour de 7%
- La frontière de décision linéaire est bien placée
- Ce taux d'erreur résiduel est normal : les classes se chevauchent
  légèrement, une droite ne peut pas les séparer parfaitement
"""

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
pertes = []
taux_erreur = []
normes = []

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


# -------------------------------------------------------
# Dataset 3 — structure non-linéaire (classes concentriques)
# -------------------------------------------------------

dataset3 = loadmat("data/reglog_data_3.mat")
X3 = dataset3["X"]
C3 = dataset3["C"]
print(X3.shape, C3.shape)

plotdata(X3, C3)

np.random.seed(0)
w, b = init()

mu3 = X3.mean(axis=1, keepdims=True)
sigma3 = X3.std(axis=1, keepdims=True)
X3_n = (X3 - mu3) / sigma3

lr = 0.01
n_iter = 1000
pertes = []
taux_erreur = []

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

"""
OBSERVATION :
- Le taux d'erreur reste très élevé (~40-50%)
- La frontière linéaire est incapable de séparer les données
- La perte ne converge pas vers une valeur satisfaisante
=> Il FAUT enrichir les features avec des termes non-linéaires !
"""

X3_phi = phi(X3_n)
print("Shape features étendues:", X3_phi.shape)

np.random.seed(0)
d = X3_phi.shape[0]
w, b = init_d(d)

lr = 0.01
n_iter = 1000
pertes = []
taux_erreur = []

for i in range(n_iter):
    pred = inference(X3_phi, w, b)
    pertes.append(objectif(pred, C3))
    pred_classe = (pred >= 0.5).astype(int)
    taux_erreur.append(np.mean(pred_classe != C3))
    dw, db = gradient_objectif(C3, pred, X3_phi)
    w = w - lr * dw
    b = b - lr * db

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pertes)
axes[0].set_title('Perte - features enrichies')
axes[0].set_xlabel('Itération')
axes[0].grid()

axes[1].plot(taux_erreur)
axes[1].set_title("Taux d'erreur")
axes[1].set_xlabel('Itération')
axes[1].grid()
plt.tight_layout()
plt.show()

print(f"Perte finale : {pertes[-1]:.4f}, Taux d'erreur : {taux_erreur[-1]*100:.2f}%")

xx, yy = np.meshgrid(np.linspace(X3[0].min() - 1, X3[0].max() + 1, 300),
                      np.linspace(X3[1].min() - 1, X3[1].max() + 1, 300))
grid = np.vstack([xx.ravel(), yy.ravel()])
grid_n = (grid - mu3) / sigma3
grid_phi = phi(grid_n)
prob_grid = inference(grid_phi, w, b).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, prob_grid, levels=50, cmap='RdYlGn', alpha=0.6)
plt.contour(xx, yy, prob_grid, levels=[0.5], colors='blue', linewidths=2)
pos = (C3 == 1).squeeze()
neg = (C3 == 0).squeeze()
plt.plot(X3[0, pos], X3[1, pos], 'g+', label='Classe 1')
plt.plot(X3[0, neg], X3[1, neg], 'r+', label='Classe 0')
plt.colorbar(label='P(c=1|x)')
plt.title('Frontière de décision – espace enrichi')
plt.legend()
plt.grid()
plt.show()

# ⬡ Neural Lab v5

> Simulateur pédagogique interactif de réseaux de neurones profonds — 100% navigateur, zéro dépendance.

![Neural Lab v5](https://img.shields.io/badge/version-v5-00e5ff?style=flat-square&labelColor=0a0c18)
![HTML](https://img.shields.io/badge/HTML-single%20file-bf5fff?style=flat-square&labelColor=0a0c18)
![License](https://img.shields.io/badge/license-MIT-00ff9d?style=flat-square&labelColor=0a0c18)
![No dependencies](https://img.shields.io/badge/deps-zero-ffd600?style=flat-square&labelColor=0a0c18)

---

## ✨ Présentation

**Neural Lab** est un simulateur visuel et pédagogique de réseaux de neurones artificiels (MLP — Multi-Layer Perceptron). Il expose **tous les calculs internes** étape par étape : forward pass, backpropagation, descente de gradient, mise à jour des poids — le tout en temps réel, dans votre navigateur, sans aucune installation.

Un seul fichier HTML. Ouvrez-le, c'est parti.

---

## 🖥️ Démo

```
git clone https://github.com/votre-pseudo/neural-lab
cd neural-lab
# Ouvrir neural_lab_v5.html dans votre navigateur
```

Ou simplement double-cliquer sur `neural_lab_v5.html`.

---

## 🚀 Fonctionnalités

### Architecture & construction
- **Constructeur simple** — définir les couches via une chaîne (`2,4,3,1`)
- **⬡ Constructeur avancé** — configurer chaque couche individuellement :
  - Nombre de neurones (slider + input)
  - Fonction d'activation propre par couche
  - Fiches d'info par activation (équation, plage, avertissements vanishing gradient)
  - Déplacer / supprimer les couches cachées
  - Preview en temps réel du réseau
  - 6 préréglages rapides (XOR, Deep, Autoencodeur, Classif 3 classes, Régression, Large)
- **Wizard** — configuration guidée par type de problème (classification binaire, multiclasse, régression)

### Fonctions d'activation (10)
`Sigmoid` · `ReLU` · `Tanh` · `Leaky ReLU` · `ELU` · `Swish/SiLU` · `GELU` · `SELU` · `Softsign` · `Linéaire` · `Softmax`

### Fonctions de perte (5)
`MSE` · `MAE` · `Log Loss (BCE)` · `Huber` · `Hinge`

### Optimiseurs (6)
`SGD` · `Momentum` · `Nesterov AG` · `RMSProp` · `Adam` · `AdamW`

### Entraînement
- **Step** — un seul exemple aléatoire
- **+Epoch** — une époque complète sur le dataset
- **×100 / ×1k / ×10k** — entraînement rapide en masse
- **Auto** — boucle continue avec vitesse réglable
- Courbe de loss en temps réel (clic pour effacer)
- Affichage Epoch / Loss / Accuracy / LR courant

### Visualisation canvas
- Réseau entièrement dessiné en Canvas 2D
- Épaisseur des connexions ∝ valeur absolue du poids
- Couleur des connexions : vert = positif, rouge = négatif
- Glow des neurones ∝ valeur d'activation
- Étiquettes : poids, biais, activations, indices
- **Légende** : entrée/caché/sortie + code couleur des connexions
- Tooltips au survol (neurone : a, z, b, activation fn ; connexion : w, type)
- Clic sur un neurone ou une connexion → panneau de détail complet

### Panneau de détail (onglet Détails)
- Pour chaque **neurone** : index, activation fn, valeur z, valeur a, biais, tableau des poids entrants avec w×a, calcul complet z = Σ(wᵢ×aᵢ) + b, dérivée f'(z), gradient local δ avec détection vanishing/exploding
- Pour chaque **connexion** : poids w, signal propagé w×a, gradient ∂L/∂w, calcul de la mise à jour

### Log des calculs
- 4 niveaux de verbosité : Complet / Moyen / Par epoch / Silencieux
- Mode **Complet** : chaque opération est détaillée (z, a, termes de la loss, deltas backprop, mises à jour)
- Copier le log dans le presse-papier

### Bibliothèque de formules (onglet Formules)
- 30+ formules référencées : activations, pertes, optimiseurs, backprop, régularisation, métriques, initialisation, architectures
- **Légende globale** des symboles mathématiques (z, a, w, b, δ, ∂L/∂w, lr, ŷ, y, σ, β₁/β₂, λ, Σ, f'(z)...)
- **Légende contextuelle** par formule (symboles pertinents uniquement)
- Graphe de la fonction pour chaque activation
- Recherche full-text + filtrage par tag
- Avantages / inconvénients pour chaque formule

### Onglet Test
- Saisir des valeurs d'entrée arbitraires
- Prédiction détaillée avec forward pass pas-à-pas
- Test complet du dataset avec accuracy par exemple

### Dataset
- 7 préréglages : XOR, AND, OR, NAND, XNOR, Half Adder, 4-bit Identity
- Éditeur de dataset visuel (ajouter/supprimer lignes)
- Import/Export JSON
- Dimensions configurables (n entrées × m sorties)

### Options avancées (onglet Options)
| Catégorie | Paramètres |
|---|---|
| Thème & couleurs | 6 thèmes (Dark, Neon, Ocean, Fire, Matrix, Pastel) + couleurs custom |
| Géométrie | Rayon neurones, épaisseur connexions, taille police, espacement H/V |
| Effets visuels | Glow, opacité, Bézier, flèches, couleur par valeur, grille, légende |
| Algorithme | Init poids (6 méthodes), momentum β, Adam β₁/β₂/ε, L2 λ, Dropout, Huber δ, gradient clipping, LR decay |
| Sauvegarde | Export JSON réseau, Import JSON, Export code Python, Export code JS |

### Redimensionnement des panneaux (v5)
- **Panneau gauche** : glisser la barre verticale (min 160px / max 520px)
- **Panneau droit** : glisser la barre verticale (min 200px / max 650px)
- **Log bas** : glisser la barre horizontale (min 60px / max 70vh)
- Double-clic sur une barre = reset aux dimensions par défaut

---

## 📐 Algorithmes implémentés

### Forward pass
```
z[l,j] = Σ(w[l,j,k] × a[l-1,k]) + b[l,j]
a[l,j] = f(z[l,j])
```

### Backpropagation (règle de la chaîne)
```
δ[L,j] = ∂L/∂a · f'(z)          ← couche sortie
δ[l,k] = (Σ δ[l+1,j] · w[l+1,j,k]) · f'(z)   ← couches cachées
∂L/∂w[l,j,k] = δ[l,j] · a[l-1,k]
∂L/∂b[l,j]   = δ[l,j]
```

### Adam (exemple)
```
m ← β₁·m + (1−β₁)·g
v ← β₂·v + (1−β₂)·g²
m̂ = m/(1−β₁ᵗ)    v̂ = v/(1−β₂ᵗ)
w ← w − lr·m̂/(√v̂ + ε)
```

---

## 🗂️ Structure du projet

```
neural-lab/
└── neural_lab_v5.html    # Application complète (fichier unique)
```

C'est volontairement un fichier unique — aucune dépendance externe sauf deux polices Google Fonts (`JetBrains Mono`, `Syne`).

---

## 💡 Expériences recommandées

| Expérience | Config suggérée |
|---|---|
| XOR non-linéairement séparable | `2,4,1` · Sigmoid · lr=0.5 · Adam |
| Vanishing gradient | `2,8,8,8,8,1` · Sigmoid · log Complet |
| ReLU vs Sigmoid (profond) | Même archi, changer activation, comparer loss |
| Divergence par lr trop élevé | lr=5 · SGD |
| Adam vs SGD | Même réseau + dataset, changer optimiseur |
| Autoencodeur | `4,2,4` · 4-bit Identity dataset |
| Activations mixtes | Constructeur avancé : ReLU cachées + Sigmoid sortie |
| Régularisation L2 | Options → λ=0.01, observer les poids |

---

## 🎨 Thèmes

| Thème | Ambiance |
|---|---|
| **Dark** | Défaut — bleu nuit profond |
| **Neon** | Cyberpunk saturé |
| **Ocean** | Bleu profond |
| **Fire** | Orange/rouge chaud |
| **Matrix** | Vert terminal |
| **Pastel** | Doux, lisible |

---

## 📦 Export

Le réseau entraîné peut être exporté en :
- **JSON** — sauvegarde complète (poids, biais, historique de loss, activations par couche)
- **Python (NumPy)** — code fonctionnel prêt à l'emploi
- **JavaScript** — code `predict()` autonome

---

## 🛠️ Technologies

- **HTML5 Canvas** — rendu du réseau et courbe de loss
- **Vanilla JS ES6+** — zéro framework, zéro build
- **CSS custom properties** — theming complet
- **Google Fonts** — JetBrains Mono · Syne

---

## 📖 Ressources pédagogiques

Ce simulateur couvre visuellement :
- Le théorème d'approximation universelle
- Le problème du vanishing/exploding gradient
- L'effet du learning rate sur la convergence
- L'initialisation des poids (Xavier, He, LeCun...)
- La régularisation (L2, Dropout)
- Les différentes familles d'optimiseurs

---

## 📄 Licence

MIT — libre d'utilisation, modification et distribution.

---

<p align="center">
  Fait avec ⬡ pour apprendre le deep learning de l'intérieur.
</p>

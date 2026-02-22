# ⬡ Neural Lab v5

> Interactive educational simulator for deep neural networks — 100% browser-based, zero dependencies.

![Neural Lab v5](https://img.shields.io/badge/version-v5-00e5ff?style=flat-square&labelColor=0a0c18)
![HTML](https://img.shields.io/badge/HTML-single%20file-bf5fff?style=flat-square&labelColor=0a0c18)
![License](https://img.shields.io/badge/license-MIT-00ff9d?style=flat-square&labelColor=0a0c18)
![No dependencies](https://img.shields.io/badge/deps-zero-ffd600?style=flat-square&labelColor=0a0c18)

---

## ✨ Overview

**Neural Lab** is a visual and educational simulator of artificial neural networks (MLP — Multi-Layer Perceptron). It exposes **all internal computations** step by step: forward pass, backpropagation, gradient descent, weight updates — all in real time, directly in your browser, with no installation required.

A single HTML file. Open it, and you're ready.

---

## 🖥️ Demo

```
git clone https://github.com/your-username/neural-lab
cd neural-lab
# Open neural_lab_v5.html in your browser
```

Or simply double-click `neural_lab_v5.html`.

---

## 🚀 Features

### Architecture & building
- **Simple builder** — define layers using a string (`2,4,3,1`)
- **⬡ Advanced builder** — configure each layer individually:
  - Number of neurons (slider + input)
  - Custom activation function per layer
  - Info cards for each activation (equation, range, vanishing gradient warnings)
  - Move / delete hidden layers
  - Real-time network preview
  - 6 quick presets (XOR, Deep, Autoencoder, 3-class classification, Regression, Large)
- **Wizard** — guided setup by problem type (binary classification, multiclass, regression)

### Activation functions (10)
`Sigmoid` · `ReLU` · `Tanh` · `Leaky ReLU` · `ELU` · `Swish/SiLU` · `GELU` · `SELU` · `Softsign` · `Linear` · `Softmax`

### Loss functions (5)
`MSE` · `MAE` · `Log Loss (BCE)` · `Huber` · `Hinge`

### Optimizers (6)
`SGD` · `Momentum` · `Nesterov AG` · `RMSProp` · `Adam` · `AdamW`

### Training
- **Step** — single random example
- **+Epoch** — one full pass over the dataset
- **×100 / ×1k / ×10k** — fast batch training
- **Auto** — continuous loop with adjustable speed
- Real-time loss curve (click to clear)
- Display of Epoch / Loss / Accuracy / current LR

### Canvas visualization
- Fully rendered network using 2D Canvas
- Connection thickness ∝ absolute weight value
- Connection color: green = positive, red = negative
- Neuron glow ∝ activation value
- Labels: weights, biases, activations, indices
- **Legend**: input/hidden/output + connection color code
- Hover tooltips (neuron: a, z, b, activation fn; connection: w, type)
- Click on neuron or connection → full detail panel

### Detail panel (Details tab)
- For each **neuron**: index, activation fn, z value, a value, bias, incoming weights table with w×a, full computation z = Σ(wᵢ×aᵢ) + b, derivative f'(z), local gradient δ with vanishing/exploding detection
- For each **connection**: weight w, propagated signal w×a, gradient ∂L/∂w, update calculation

### Computation log
- 4 verbosity levels: Full / Medium / Per epoch / Silent
- **Full mode**: every operation is detailed (z, a, loss terms, backprop deltas, updates)
- Copy log to clipboard

### Formula library (Formulas tab)
- 30+ referenced formulas: activations, losses, optimizers, backprop, regularization, metrics, initialization, architectures
- **Global legend** of mathematical symbols (z, a, w, b, δ, ∂L/∂w, lr, ŷ, y, σ, β₁/β₂, λ, Σ, f'(z)...)
- **Contextual legend** per formula (only relevant symbols)
- Graph of each activation function
- Full-text search + tag filtering
- Pros / cons for each formula

### Test tab
- Enter arbitrary input values
- Detailed prediction with step-by-step forward pass
- Full dataset testing with per-example accuracy

### Dataset
- 7 presets: XOR, AND, OR, NAND, XNOR, Half Adder, 4-bit Identity
- Visual dataset editor (add/remove rows)
- Import/Export JSON
- Configurable dimensions (n inputs × m outputs)

### Advanced options (Options tab)
| Category | Settings |
|---|---|
| Theme & colors | 6 themes (Dark, Neon, Ocean, Fire, Matrix, Pastel) + custom colors |
| Geometry | Neuron radius, connection thickness, font size, H/V spacing |
| Visual effects | Glow, opacity, Bézier, arrows, color by value, grid, legend |
| Algorithm | Weight init (6 methods), momentum β, Adam β₁/β₂/ε, L2 λ, Dropout, Huber δ, gradient clipping, LR decay |
| Save | Export network JSON, Import JSON, Export Python code, Export JS code |

### Panel resizing (v5)
- **Left panel**: drag vertical bar (min 160px / max 520px)
- **Right panel**: drag vertical bar (min 200px / max 650px)
- **Bottom log**: drag horizontal bar (min 60px / max 70vh)
- Double-click a bar = reset to default size

---

## 📐 Implemented algorithms

### Forward pass
```
z[l,j] = Σ(w[l,j,k] × a[l-1,k]) + b[l,j]
a[l,j] = f(z[l,j])
```

### Backpropagation (chain rule)
```
δ[L,j] = ∂L/∂a · f'(z)
δ[l,k] = (Σ δ[l+1,j] · w[l+1,j,k]) · f'(z)
∂L/∂w[l,j,k] = δ[l,j] · a[l-1,k]
∂L/∂b[l,j]   = δ[l,j]
```

### Adam (example)
```
m ← β₁·m + (1−β₁)·g
v ← β₂·v + (1−β₂)·g²
m̂ = m/(1−β₁ᵗ)    v̂ = v/(1−β₂ᵗ)
w ← w − lr·m̂/(√v̂ + ε)
```

---

## 📄 License

MIT — free to use, modify, and distribute.

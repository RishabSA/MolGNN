# MolGNN

MolGNN is a project for molecular solubility prediction using Graph Neural Networks trained on the **ESOL** dataset from **MoleculeNet**. The goal is to predict log solubility values (mol/L) directly from molecular graph structure encoded as vectors from **SMILES** sequence strings.

## Models

Five graph neural network (GNN) architectures were implemented and trained. All models share a common design: residual (skip) connections across message-passing layers with constant hidden width, global mean pooling for graph-level readout, and a 2-layer MLP output head.

| Model                 | Conv Layer        | Edge Features | Activation | Description                                                                        |
| --------------------- | ----------------- | ------------- | ---------- | ---------------------------------------------------------------------------------- |
| **GCN**               | `GCNConv`         | No            | ReLU       | Spectral graph convolutions with symmetric normalization                           |
| **GAT**               | `GATConv`         | No            | ELU        | Multi-head attention over node neighborhoods                                       |
| **GIN**               | `GINConv`         | No            | ReLU       | Maximally expressive under the WL isomorphism test; 2-layer MLP per conv           |
| **GATv2**             | `GATv2Conv`       | Yes           | ELU        | Dynamic attention (strictly more expressive than GAT); uses bond features          |
| **Graph Transformer** | `TransformerConv` | Yes           | ReLU       | Full transformer-style attention with learnable skip gate (beta) and bond features |

## Results

Evaluated on an 80/20 train/test split of ESOL (1128 molecules). All models trained with AdamW optimizer and MSE loss.

| Model             | Layers | Hidden | Heads | LR   | Epochs | Test MAE (mol/L) |
| ----------------- | ------ | ------ | ----- | ---- | ------ | ---------------- |
| **GCN**           | 3      | 128    | -     | 1e-3 | 400    | **0.4583**       |
| GAT               | 3      | 128    | 4     | 1e-3 | 400    | 0.5364           |
| Graph Transformer | 3      | 128    | 4     | 5e-4 | 400    | 0.5395           |
| GATv2             | 3      | 128    | 4     | 5e-4 | 400    | 0.5597           |
| GIN               | 4      | 128    | -     | 1e-3 | 500    | 0.6979           |

## Setup

```bash
git clone https://github.com/RishabSA/MolGNN.git
cd MolGNN
pip install -r requirements.txt
```

## Usage

### Streamlit App

Run the interactive web app built with streamlit to predict log solubility from any SMILES string:

```bash
streamlit run app.py
```

The app loads the pre-trained GCN model, renders molecules in real time using RDKit, and displays the predicted log solubility on button click.

### Training

Open `molecule_gnn.ipynb` to train all five models using PyTorch and PyTorch Geometric. The ESOL dataset is downloaded automatically using PyTorch Geometric. Each model section includes instantiation, training, evaluation, and an example prediction.

## Dataset

**ESOL** (Estimated SOLubility) contains 1128 small organic molecules with experimentally measured log solubility values in mol/L. Node features (9-dim) encode atom properties (atomic number, chirality, degree, etc.) and edge features (3-dim) encode bond properties (bond type, stereo, aromaticity). The dataset is loaded via PyTorch Geometric's `MoleculeNet` interface and cached locally under `data/`.

## Architecture Details

Each model follows the same macro-architecture:

1. **Input head** — single conv layer projecting 9-dim node features to `hidden_channels`
2. **Message-passing layers** — `num_layers` conv blocks at constant width with BatchNorm, activation, dropout, and **residual connections** (`x = x + layer(x)`)
3. **Global mean pooling** — aggregates node embeddings into a single graph-level vector
4. **Output MLP** — `Linear(hidden, hidden/2) -> Activation -> Dropout -> Linear(hidden/2, 1)`

Edge-aware models, such as GATv2 and Graph Transformer are are also trained, and additionally receive `edge_attr` through the conv layers for edge features.

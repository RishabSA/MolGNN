import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from scripts.models import GraphConvolutionalNetwork, load_model
from scripts.inference import predict_log_solubility_from_smiles

st.set_page_config(page_title="MolGNN", page_icon=":test_tube:", layout="centered")

example_molecules = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Acetic Acid": "CC(=O)O",
    "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}


@st.cache_resource
def get_model():
    device = torch.device("cpu")
    model = GraphConvolutionalNetwork(
        num_layers=3,
        in_channels=9,
        hidden_channels=128,
        out_channels=1,
        dropout=0.1,
    )
    model = load_model(model, path="models/gcn_model.pt", device=device)

    return model, device


model, device = get_model()

st.title(":test_tube: MolGNN")
st.markdown(
    "Predict the **log solubility** (mol/L) of a molecule from its SMILES sequence representation using a Graph Convolutional Network (GCN) trained on the ESOL dataset from MoleculeNet."
)

col_input, col_examples = st.columns([3, 2], gap="medium")

with col_input:
    smiles_input = st.text_input(
        "SMILES sequence",
        value="CCO",
        placeholder="e.g. CCO, c1ccccc1, CC(=O)O",
    )

with col_examples:
    selected = st.selectbox(
        "Quick examples",
        options=list(example_molecules.keys()),
        index=None,
        placeholder="Choose an example...",
    )

    if selected:
        smiles_input = example_molecules[selected]

molecule = Chem.MolFromSmiles(smiles_input) if smiles_input else None

if smiles_input and molecule is None:
    st.error("Invalid SMILES sequence. Please check the input and try again.")

if molecule is not None:
    img = Draw.MolToImage(molecule, size=(400, 400))
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.image(img, use_container_width=True)

if st.button(
    "Predict Log Solubility (mol/L)",
    type="primary",
    use_container_width=True,
    disabled=(molecule is None),
):
    with st.spinner("Running GNN inference..."):
        prediction = predict_log_solubility_from_smiles(
            model=model, smiles_seq=smiles_input, device=device
        )

    col_metric, col_info = st.columns(2)
    with col_metric:
        st.metric(label="Predicted Log Solubility", value=f"{prediction:.4f} mol/L")
    with col_info:
        st.markdown(
            f"**SMILES sequence:** `{smiles_input}` | **Atoms:** {molecule.GetNumAtoms()} | **Bonds:** {molecule.GetNumBonds()}"
        )

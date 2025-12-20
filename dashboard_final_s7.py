 # ===============================================================
#        DASHBOARD IA MÉDICALE – EXPLICABILITÉ
# ===============================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image



st.set_page_config(
    page_title=" Diagnostic Médical IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title(" Navigation")

menu = st.sidebar.radio(
    "Aller vers :",
    [" Accueil", " Prédiction (IA)", " Analyse"]
)


# OpenCV (fallback si non installé)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ===============================================================
# 1. MODÈLE (EfficientNet V2)
# ===============================================================

class MedNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.m = efficientnet_v2_s(weights=weights)

        # Adapter 1 canal → 3 canaux
        self.m.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=3, stride=1, padding=1, bias=False
        )

        in_features = self.m.classifier[1].in_features
        self.m.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 14)
        )

    def forward(self, x):
        return self.m(x)

# ===============================================================
# 2. GRADCAM++
# ===============================================================

class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model
        self.grad = None
        self.act = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.act = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def generate(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()

        grads = self.grad
        acts = self.act

        weights = grads.pow(2)
        alpha = weights / (2 * weights + torch.sum(acts * grads, dim=(2, 3), keepdim=True))
        cam = torch.sum(alpha * F.relu(grads) * acts, dim=1).squeeze().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam /= cam.max() + 1e-12
        return cam

# ===============================================================
# 3. OCCLUSION SENSITIVITY
# ===============================================================

def occlusion_sensitivity(model, img_tensor, patch=20):
    model.eval()
    _, _, H, W = img_tensor.shape
    heatmap = np.zeros((H, W))

    base_prob = torch.sigmoid(model(img_tensor)).detach().numpy()[0].max()

    for i in range(0, H, patch):
        for j in range(0, W, patch):
            img_occ = img_tensor.clone()
            img_occ[:, :, i:i+patch, j:j+patch] = 0
            prob = torch.sigmoid(model(img_occ)).detach().numpy()[0].max()
            heatmap[i:i+patch, j:j+patch] = base_prob - prob

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-12
    return heatmap

# ===============================================================
# 4. VISUALISATION (SANS CV2 SI BESOIN)
# ===============================================================
# ===============================================================
# NORMALISATION IMAGE
# ===============================================================

def normalize_img(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-12)

def overlay_heatmap(img, heatmap):
    img = normalize_img(img)

    # Resize heatmap 14x14 → 224x224
    if heatmap.shape != img.shape:
        heatmap = np.array(
            Image.fromarray(heatmap).resize((224, 224))
        )

    heatmap = normalize_img(heatmap)

    overlay = np.stack([
        img * 0.6 + heatmap * 0.4,
        img * 0.6,
        img * 0.6
    ], axis=-1)

    return np.uint8(overlay * 255)


# ===============================================================
# 5. STREAMLIT APP
# ===============================================================

# ===============================================================
# 5. STREAMLIT APP (AVEC NAVIGATION)
# ===============================================================

# ---------- PAGE ACCUEIL ----------
if menu == " Accueil":
    st.title(" Diagnostic Médical Assisté par IA")

    st.markdown("""
    ### Bienvenue 👋  
    Cette application utilise un **modèle d’intelligence artificielle (IA)**  pour analyser des radiographies thoraciques et fournir des  **explications visuelles** (GradCAM++, Occlusion Sensitivity).

    -> Allez dans **Prédiction (IA)** pour importer une image.
    """)

# ---------- PAGE PRÉDICTION (TON DASHBOARD COMPLET) ----------
elif menu == " Prédiction (IA)":
    st.title(" IA Médicale – Explicabilité Radiologique")
    st.write("GradCAM++ et Occlusion Sensitivity sur radiographies thoraciques")

    # Charger modèle
    model = MedNet()
    model_path = "models/best_week6.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        st.success("✅ Modèle entraîné chargé")
    else:
        st.warning("⚠️ Modèle non trouvé — poids aléatoires utilisés")

    model.eval()

    uploaded_file = st.file_uploader(
        " Importer une radiographie thoracique",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("L").resize((224, 224))
        img_np = np.array(img) / 255.0
        img_tensor = torch.tensor(img_np).float().unsqueeze(0).unsqueeze(0)

        st.subheader(" Radiographie originale")
        st.image(img, width=300)

        # Prédiction
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits)[0].numpy()

        class_names = [
            "Aucune anomalie", "Atélectasie", "Cardiomégalie", "Consolidation",
            "Œdème", "Épanchement pleural", "Emphysème", "Fibrose", "Hernie",
            "Infiltration", "Masse", "Nodule", "Pneumonie", "Pneumothorax"
        ]

        top_idx = int(np.argmax(probs))
        top_prob = probs[top_idx] * 100

        st.subheader("📌 Résultat du diagnostic")

        if top_idx == 0:
            st.success(f"✅ Patient probablement sain ({top_prob:.1f} %)")
        else:
            st.error(f"⚠️ Pathologie suspectée : **{class_names[top_idx]}**")
            st.write(f"Probabilité estimée : **{top_prob:.1f} %**")

        st.subheader(" Scores détaillés (triés)")

        # Créer un DataFrame avec les scores
        scores_df = pd.DataFrame({
            "Pathologie": class_names,
            "Probabilité (%)": probs * 100
        })

        # Trier du plus élevé au plus faible
        scores_df = scores_df.sort_values(
            by="Probabilité (%)",
            ascending=False
        ).reset_index(drop=True)

        st.caption("🔴 Risque élevé | 🟡 Risque modéré | 🟢 Faible risque")

        # Affichage coloré
        for idx, row in scores_df.iterrows():
            label = row["Pathologie"]
            value = row["Probabilité (%)"]

            if idx < 3 and value >= 50:
                st.error(f"🔴 {label} : {value:.1f} %")
            elif value >= 50:
                st.warning(f"🟡 {label} : {value:.1f} %")
            else:
                st.success(f"🟢 {label} : {value:.1f} %")


        # GradCAM++
        st.subheader(" GradCAM++")
        target_layer = model.m.features[-1]
        gradcam = GradCAMPP(model, target_layer)
        cam = gradcam.generate(img_tensor, top_idx)
        overlay_cam = overlay_heatmap(img_np, cam)
        st.image(overlay_cam, width=300)

        # Occlusion
        st.subheader(" Occlusion Sensitivity")
        occ = occlusion_sensitivity(model, img_tensor)
        overlay_occ = overlay_heatmap(img_np, occ)
        st.image(overlay_occ, width=300)

# ---------- PAGE ANALYSE ----------
# ---------- PAGE ANALYSE ----------
elif menu == " Analyse":
    st.title(" Analyse globale du modèle")

    st.markdown("###  Visualisation graphique")

    # Données exemple (à remplacer plus tard par vraies stats si besoin)
    df = pd.DataFrame({
        "Classe": [
            "Aucune anomalie", "Atélectasie", "Cardiomégalie", "Consolidation",
            "Œdème", "Épanchement pleural", "Emphysème", "Fibrose",
            "Hernie", "Infiltration", "Masse", "Nodule",
            "Pneumonie", "Pneumothorax"
        ],
        "Probabilité moyenne (%)": [
            62, 18, 12, 15, 20, 25, 10, 14,
            5, 22, 9, 11, 30, 8
        ]
    })

    # ---------- DASHBOARD VISUEL ----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("####  Distribution des probabilités")
        st.bar_chart(
            df.set_index("Classe")["Probabilité moyenne (%)"]
        )

    with col2:
        st.markdown("####  Tableau récapitulatif")
        st.dataframe(
            df.sort_values("Probabilité moyenne (%)", ascending=False),
            use_container_width=True
        )

    st.info(
        "Ces graphiques permettent d’analyser le comportement global du modèle "
        "et la distribution moyenne des pathologies détectées."
    )
    # =====================================================
    # DASHBOARD SAIN vs MALADE
    # =====================================================

    st.markdown("###  Dashboard Patient Sain vs Malade")

    # Données synthétiques (exemple)
    df_health = pd.DataFrame({
        "État du patient": ["Sain", "Malade"],
        "Proportion (%)": [45, 55]
    })

    col3, col4 = st.columns(2)

    # ---- Indicateurs clés ----
    with col3:
        st.metric(
            label="🟢 Patients sains",
            value="45 %",
            delta="-5 %"
        )

    with col4:
        st.metric(
            label="🔴 Patients malades",
            value="55 %",
            delta="+5 %"
        )

    # ---- Graphique comparatif ----
    st.markdown("####  Comparaison globale")

    st.bar_chart(
        df_health.set_index("État du patient")
    )

    st.success(
        "Ce dashboard permet une lecture rapide de la proportion globale "
        "des patients sains et malades détectés par le modèle."
    )

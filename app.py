
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Iris Data Explorer", layout="centered")

st.title("🌸 Analyse du Dataset Iris")
st.markdown("""
Cette application permet d'analyser les caractéristiques des fleurs d'Iris.
**Auteur :Daniel**
""")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # ATTENTION : J'ai mis sep=";" car ton fichier utilise des points-virgules
    df = pd.read_csv("Iris.csv", sep=";")
    return df

try:
    df = load_data()
    st.success("Données chargées avec succès !")
except FileNotFoundError:
    st.error("Fichier Iris.csv introuvable !")
    st.stop()

# --- PARTIE 1 : NETTOYAGE & APERÇU ---
st.header("1. Exploration et Nettoyage")

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Dimensions :** {df.shape[0]} lignes, {df.shape[1]} colonnes")
with col2:
    st.write("**Colonnes :**", list(df.columns))

if st.checkbox("Afficher les 5 premières lignes"):
    st.dataframe(df.head())

# Vérification des valeurs manquantes
missing_val = df.isnull().sum().sum()
st.write(f"**Valeurs manquantes :** {missing_val}")

# Gestion des doublons
duplicates = df.duplicated().sum()
st.write(f"**Doublons détectés :** {duplicates}")

if duplicates > 0:
    if st.button("Supprimer les doublons"):
        df = df.drop_duplicates()
        st.success(f"Doublons supprimés ! Nouvelles dimensions : {df.shape}")

# --- PARTIE 2 : STATISTIQUES ---
st.header("2. Statistiques Descriptives")
st.write(df.describe())

# --- PARTIE 3 : VISUALISATION ---
st.header("3. Visualisation Interactive")

# Selectbox pour choisir les axes
st.subheader("Graphique de dispersion (Scatter Plot)")
axis_options = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

x_axis = st.selectbox("Choisis l'axe X", axis_options, index=0)
y_axis = st.selectbox("Choisis l'axe Y", axis_options, index=2)

# Création du graphique
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Species', ax=ax)
plt.title(f"{x_axis} vs {y_axis}")
st.pyplot(fig)

# Histogramme
st.subheader("Distribution des variables")
hist_col = st.selectbox("Choisis une colonne pour l'histogramme", axis_options)
fig2, ax2 = plt.subplots()
sns.histplot(data=df, x=hist_col, kde=True, hue="Species", element="step", ax=ax2)
st.pyplot(fig2)
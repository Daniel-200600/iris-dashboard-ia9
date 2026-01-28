
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
# Options communes
axis_options = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

# --- Menu déroulant pour le design / thème ---
st.subheader("Apparence")
design = st.selectbox(
    "Choisis un design (thème)",
    ['Seaborn darkgrid', 'Seaborn whitegrid', 'Seaborn ticks', 'Matplotlib ggplot', 'Matplotlib classic'],
    index=0,
)

if design.startswith('Seaborn'):
    # applique le style seaborn
    _, sns_style = design.split(' ', 1)
    sns.set_theme(style=sns_style)
else:
    mpl_style = 'ggplot' if 'ggplot' in design else 'classic'
    plt.style.use(mpl_style)

# --- Filtre des espèces (menu déroulant multi-sélection) ---
st.subheader("Filtrer les données")
species_options = df['Species'].unique().tolist()
selected_species = st.multiselect("Sélectionner les espèces (laisser vide = toutes)", options=species_options, default=species_options)

if len(selected_species) == 0:
    filtered_df = df.copy()
else:
    filtered_df = df[df['Species'].isin(selected_species)]

# --- Choix du type de graphique ---
st.header("Visualisations")
plot_choice = st.selectbox("Type de graphique", ['Scatter', 'Boxplot', 'Histogram', 'Pairplot'])

if plot_choice == 'Scatter':
    st.subheader("Graphique de dispersion (Scatter Plot)")
    x_axis = st.selectbox("Choisis l'axe X", axis_options, index=0, key='scatter_x')
    y_axis = st.selectbox("Choisis l'axe Y", axis_options, index=2, key='scatter_y')

    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue='Species', ax=ax)
    ax.set_title(f"{x_axis} vs {y_axis}")
    st.pyplot(fig)

elif plot_choice == 'Boxplot':
    st.subheader("Boxplot par espèce")
    box_col = st.selectbox("Choisis une colonne numérique pour le boxplot", axis_options, key='box_col')
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x='Species', y=box_col, ax=ax)
    # ajoute les points individuels pour plus de lisibilité
    sns.stripplot(data=filtered_df, x='Species', y=box_col, color='0.3', size=4, jitter=True, ax=ax)
    ax.set_title(f"Boxplot de {box_col} par espèce")
    st.pyplot(fig)

elif plot_choice == 'Histogram':
    st.subheader("Distribution des variables")
    hist_col = st.selectbox("Choisis une colonne pour l'histogramme", axis_options, key='hist_col')
    fig2, ax2 = plt.subplots()
    sns.histplot(data=filtered_df, x=hist_col, kde=True, hue='Species', element='step', ax=ax2)
    ax2.set_title(f"Histogramme de {hist_col}")
    st.pyplot(fig2)

elif plot_choice == 'Pairplot':
    st.subheader("Pairplot (nuages de points matriciels)")
    cols = st.multiselect("Choisis les colonnes à inclure", axis_options, default=axis_options)
    if len(cols) < 2:
        st.warning("Choisir au moins 2 colonnes pour le pairplot.")
    else:
        # pairplot retourne un PairGrid ; afficher sa figure
        pairgrid = sns.pairplot(filtered_df[cols + ['Species']], hue='Species', corner=True)
        st.pyplot(pairgrid.fig)

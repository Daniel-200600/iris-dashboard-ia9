import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# sklearn pour le modèle de prédiction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Iris Data Explorer", layout="wide")

st.title("🌸 Analyse du Dataset Iris")
st.markdown("""
Cette application permet d'analyser les caractéristiques des fleurs d'Iris.
**Auteur : Daniel**
""")

# --- THEME & COULEURS (thème clair et agréable) ---
# Couleurs douces et lumineuses pour une meilleure lisibilité
PRIMARY = '#5DADE2'      # bleu ciel doux
SECONDARY = '#48C9B0'    # turquoise décontractant
ACCENT = '#F5B041'       # orange doux
BG_MAIN = '#FAFAFA'      # fond blanc cassé
CARD_BG = '#FFFFFF'      # cartes blanches
SIDEBAR_BG = '#E8E8E8'   # fond de la sidebar (gris clair plus sombre)
# Couleurs de texte
TEXT_COLOR = '#2C3E50'   # gris foncé pour le texte principal
SIDEBAR_TEXT = '#2C3E50' # texte de la sidebar

# Palette par espèce (couleurs agréables et distinctes)
SPECIES_PALETTE = {
    'Iris-setosa': '#5DADE2',      # bleu ciel
    'Iris-versicolor': '#48C9B0',  # turquoise
    'Iris-virginica': '#F5B041',   # orange doux
}


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

# ------------------ SIDEBAR (Navigation & Options) ------------------
# CSS global pour enlever les fonds blancs et appliquer les couleurs vives
st.markdown(
    f"""
    <style>
    /* Force a pure black background across common Streamlit containers */
    html, body, [data-testid="stAppViewContainer"], .stApp, .block-container {{
        background-color: {BG_MAIN} !important;
    }}
    /* Texte principal (zone de contenu) */
    .main, .main * {{
        color: {TEXT_COLOR} !important;
    }}

    /* Main content cards / sections */
    .css-1d391kg, .css-1v3fvcr, .css-10trblm, .stCard {{
        background-color: {CARD_BG} !important;
        color: {TEXT_COLOR} !important;
    }}

/* Sidebar styling (texte laissé en blanc pour lisibilité) */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {SIDEBAR_BG} !important;
        color: {SIDEBAR_TEXT} !important;
    }}
    [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div{{
        color: {SIDEBAR_TEXT} !important;
    }}

    /* Buttons */
    div.stButton > button {{
        background-color: {PRIMARY} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {SECONDARY} !important;
    }}

    /* Tables and dataframes */
    .stDataFrame div, .css-1ex1afd, .css-1v0mbdj {{
        background-color: {CARD_BG} !important;
        color: {TEXT_COLOR} !important;
    }}

    /* Headings / expanders */
    .streamlit-expanderHeader, .css-10trblm h1, .css-10trblm h2 {{
        color: {TEXT_COLOR} !important;
    }}

    /* Inputs & widgets fallback */
    .stSelectbox > div, .stTextInput > div, .stNumberInput > div, .stSlider > div {{
        background-color: transparent !important;
        color: {TEXT_COLOR} !important;
    }}

    /* Make SVGs and canvases match the dark background */
    svg, canvas {{
        background-color: transparent !important;
    }}

    /* Generic fallback only for main content (avoid overriding sidebar) */
    .main, .main * {{
        color: {TEXT_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation principale (barre en haut)
nav_options = ['Accueil', 'Exploration', 'Visualisations', 'Modèle', 'Prédiction', 'Données', 'Paramètres', 'À propos']

# Render a simple horizontal navigation bar using buttons. Clicking a button stores the
# selected page in session_state['nav']. The sidebar remains available for filters/options.
cols_nav = st.columns(len(nav_options))
for i, opt in enumerate(nav_options):
    if cols_nav[i].button(opt):
        st.session_state['nav'] = opt

nav = st.session_state.get('nav', nav_options[0])

# Filtres et options globales dans la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")
species_options = df['Species'].unique().tolist()
selected_species = st.sidebar.multiselect("Espèces", options=species_options, default=species_options)

st.sidebar.markdown("---")
st.sidebar.subheader("Apparence")
design = st.sidebar.selectbox(
    "Choisis un design (thème)",
    ['Seaborn darkgrid', 'Seaborn ticks', 'Matplotlib ggplot', 'Matplotlib classic'],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Modèle (rapide)")
model_choice = st.sidebar.selectbox("Choisis un modèle", ['RandomForest', 'LogisticRegression'])
rf_estimators = None
if model_choice == 'RandomForest':
    rf_estimators = st.sidebar.slider("n_estimators", min_value=10, max_value=300, value=100, step=10)

# Appliquer le thème choisi (mais forcer les couleurs vives et fonds sombre)
if design.startswith('Seaborn'):
    _, sns_style = design.split(' ', 1)
    # On applique le style Seaborn choisi puis on force la palette
    sns.set_theme(style=sns_style, palette=list(SPECIES_PALETTE.values()))
else:
    mpl_style = 'ggplot' if 'ggplot' in design else 'classic'
    plt.style.use(mpl_style)

# Forcer des rcParams pour éviter les fonds blancs et appliquer nos couleurs
plt.rcParams.update({
    'figure.facecolor': BG_MAIN,
    'axes.facecolor': CARD_BG,
    'savefig.facecolor': BG_MAIN,
    'axes.edgecolor': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'axes.titlecolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'grid.color': '#3b1f12',
    'legend.facecolor': CARD_BG,
    'legend.edgecolor': TEXT_COLOR,
    'axes.titleweight': 'bold',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

# Préparer un cmap personnalisé pour la heatmap
HEATMAP_CMAP = LinearSegmentedColormap.from_list('iris_cmap', [SECONDARY, ACCENT, PRIMARY])

# Filtrer les données globalement
if len(selected_species) == 0:
    filtered_df = df.copy()
else:
    filtered_df = df[df['Species'].isin(selected_species)]

# Options d'axes utilisées par plusieurs pages
axis_options = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']


@st.cache_resource
def train_model(df, model_name='RandomForest', n_estimators=100, random_state=42):
    X = df[axis_options].values
    y = df['Species'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc
    )

    if model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        model = LogisticRegression(max_iter=500, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'model': model,
        'le': le,
        'scaler': scaler,
        'acc': acc,
        'report': report,
        'cm': cm,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
    }


# --- RENDERING PAR PAGES SELON LA NAVIGATION ---
if nav == 'Accueil':
    st.header("Bienvenue")
    st.write("Utilisez la barre de navigation à gauche pour naviguer entre les pages : Exploration, Visualisations, Modèle, Prédiction, etc.")
    if st.checkbox("Afficher un aperçu rapide des données"):
        st.dataframe(df.head())

elif nav == 'Exploration':
    st.header("1. Exploration et Nettoyage")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dimensions :** {df.shape[0]} lignes, {df.shape[1]} colonnes")
    with col2:
        st.write("**Colonnes :**", list(df.columns))

    if st.checkbox("Afficher les 5 premières lignes", key='explore_head'):
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

    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df.describe())

elif nav == 'Visualisations':
    st.header("Visualisations")
    st.write("Les filtres et le thème sont disponibles dans la barre de navigation à gauche.")

    plot_choice = st.selectbox("Type de graphique", ['Scatter', 'Boxplot', 'Histogram', 'Pairplot'], key='plot_choice')

    if plot_choice == 'Scatter':
        st.subheader("Graphique de dispersion (Scatter Plot)")
        x_axis = st.selectbox("Choisis l'axe X", axis_options, index=0, key='scatter_x')
        y_axis = st.selectbox("Choisis l'axe Y", axis_options, index=2, key='scatter_y')

        fig, ax = plt.subplots()
        # Palette spécifique pour le scatter (couleurs vives par espèce)
        scatter_palette = {k: SPECIES_PALETTE.get(k, PRIMARY) for k in filtered_df['Species'].unique()}
        sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, hue='Species', palette=scatter_palette, s=80, edgecolor=TEXT_COLOR, linewidth=0.6, ax=ax)
        ax.set_title(f"{x_axis} vs {y_axis}", color=TEXT_COLOR)
        st.pyplot(fig)

    elif plot_choice == 'Boxplot':
        st.subheader("Boxplot par espèce")
        box_col = st.selectbox("Choisis une colonne numérique pour le boxplot", axis_options, key='box_col')
        fig, ax = plt.subplots()
        # Utiliser une palette dégradée pour les boxplots
        box_palette = [SPECIES_PALETTE.get(s, PRIMARY) for s in filtered_df['Species'].unique()]
        sns.boxplot(data=filtered_df, x='Species', y=box_col, palette=box_palette, ax=ax, fliersize=3)
        sns.stripplot(data=filtered_df, x='Species', y=box_col, color=TEXT_COLOR, size=4, jitter=True, ax=ax, edgecolor='none')
        ax.set_title(f"Boxplot de {box_col} par espèce", color=TEXT_COLOR)
        st.pyplot(fig)

    elif plot_choice == 'Histogram':
        st.subheader("Distribution des variables")
        hist_col = st.selectbox("Choisis une colonne pour l'histogramme", axis_options, key='hist_col')
        fig2, ax2 = plt.subplots()
        # Histogramme avec transparence et palette par espèce
        hist_palette = [SPECIES_PALETTE.get(s, PRIMARY) for s in filtered_df['Species'].unique()]
        sns.histplot(data=filtered_df, x=hist_col, kde=True, hue='Species', element='step', palette=hist_palette, alpha=0.6, ax=ax2)
        ax2.set_title(f"Histogramme de {hist_col}", color=TEXT_COLOR)
        st.pyplot(fig2)

    elif plot_choice == 'Pairplot':
        st.subheader("Pairplot (nuages de points matriciels)")
        cols = st.multiselect("Choisis les colonnes à inclure", axis_options, default=axis_options)
        if len(cols) < 2:
            st.warning("Choisir au moins 2 colonnes pour le pairplot.")
        else:
            # Palette plus soft pour le pairplot mais qui reste dans le thème
            pair_palette = {k: SPECIES_PALETTE.get(k, PRIMARY) for k in filtered_df['Species'].unique()}
            pairgrid = sns.pairplot(filtered_df[cols + ['Species']], hue='Species', corner=True, palette=pair_palette, plot_kws={'edgecolor': TEXT_COLOR, 's': 40})
            pairgrid.fig.patch.set_facecolor(BG_MAIN)
            st.pyplot(pairgrid.fig)

elif nav == 'Modèle':
    st.header("Modèle de prédiction")
    st.write("Sélectionnez les options dans la barre latérale puis entraînez le modèle.")

    # Entraînement via bouton pour éviter ré-entraînement automatique
    if st.button('Entraîner le modèle'):
        with st.spinner('Entraînement du modèle...'):
            trained = train_model(df, model_name=model_choice, n_estimators=rf_estimators or 100)
            st.session_state['trained'] = trained
            st.success('Entraînement terminé.')

    trained = st.session_state.get('trained')
    if trained:
        st.subheader("Performance du modèle")
        st.write(f"Accuracy (test) : {trained['acc']:.3f}")
        st.text("Rapport de classification :")
        st.text(trained['report'])

        fig_cm, ax_cm = plt.subplots()
    # Heatmap avec cmap personnalisé (marron -> jaune)
        sns.heatmap(trained['cm'], annot=True, fmt='d', cmap=HEATMAP_CMAP, ax=ax_cm,
                    xticklabels=trained['le'].classes_, yticklabels=trained['le'].classes_, cbar_kws={'label': 'count'})
        ax_cm.set_xlabel('Prédit')
        ax_cm.set_ylabel('Vrai')
        ax_cm.set_title('Matrice de confusion')
        st.pyplot(fig_cm)
    else:
        st.info("Aucun modèle entraîné. Cliquez sur 'Entraîner le modèle' pour lancer l'entraînement.")

elif nav == 'Prédiction':
    st.header("Prédire l'espèce d'une fleur")
    st.write("Renseigne les caractéristiques ci-dessous puis clique sur 'Prédire' pour obtenir l'espèce estimée.")

    # Définir les bornes des sliders à partir du jeu de données
    input_sep_len_min, input_sep_len_max = float(df['SepalLength'].min()), float(df['SepalLength'].max())
    input_sep_wid_min, input_sep_wid_max = float(df['SepalWidth'].min()), float(df['SepalWidth'].max())
    input_pet_len_min, input_pet_len_max = float(df['PetalLength'].min()), float(df['PetalLength'].max())
    input_pet_wid_min, input_pet_wid_max = float(df['PetalWidth'].min()), float(df['PetalWidth'].max())

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        s_len = st.slider('SepalLength', min_value=input_sep_len_min, max_value=input_sep_len_max, value=float(df['SepalLength'].median()), step=0.1)
    with col_b:
        s_wid = st.slider('SepalWidth', min_value=input_sep_wid_min, max_value=input_sep_wid_max, value=float(df['SepalWidth'].median()), step=0.1)
    with col_c:
        p_len = st.slider('PetalLength', min_value=input_pet_len_min, max_value=input_pet_len_max, value=float(df['PetalLength'].median()), step=0.1)
    with col_d:
        p_wid = st.slider('PetalWidth', min_value=input_pet_wid_min, max_value=input_pet_wid_max, value=float(df['PetalWidth'].median()), step=0.1)

    trained = st.session_state.get('trained')
    if not trained:
        st.warning("Aucun modèle entraîné. Allez à la page 'Modèle' pour entraîner un modèle, ou entraînez rapidement ci-dessous.")
        if st.button('Entraîner rapidement ici'):
            with st.spinner('Entraînement rapide...'):
                trained = train_model(df, model_name=model_choice, n_estimators=rf_estimators or 100)
                st.session_state['trained'] = trained
                st.success('Entraînement terminé.')

    if st.button('Prédire'):
        trained = st.session_state.get('trained')
        if not trained:
            st.error("Aucun modèle disponible. Entraînez un modèle d'abord.")
        else:
            model = trained['model']
            le = trained['le']
            scaler = trained['scaler']

            X_new = [[s_len, s_wid, p_len, p_wid]]
            X_new_scaled = scaler.transform(X_new)

            y_new_pred_idx = model.predict(X_new_scaled)
            y_new_pred = le.inverse_transform(y_new_pred_idx)[0]

            st.success(f"Espèce prédite : {y_new_pred}")

            # Probabilités si disponibles
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_new_scaled)[0]
                prob_series = pd.Series(probs, index=le.classes_).sort_values(ascending=False)
                st.write("Probabilités :")
                st.table(prob_series)

            # Afficher le point prédit sur un scatter si l'utilisateur sélectionne 2 axes
            if 'scatter_x' in st.session_state and 'scatter_y' in st.session_state:
                try:
                    fig_pred, ax_pred = plt.subplots()
                    sns.scatterplot(data=filtered_df, x=st.session_state['scatter_x'], y=st.session_state['scatter_y'], hue='Species', ax=ax_pred)
                    # choisir les bonnes coordonnées selon l'axe
                    xval = s_len if st.session_state['scatter_x'] == 'SepalLength' else s_wid if st.session_state['scatter_x'] == 'SepalWidth' else p_len if st.session_state['scatter_x'] == 'PetalLength' else p_wid
                    yval = s_len if st.session_state['scatter_y'] == 'SepalLength' else s_wid if st.session_state['scatter_y'] == 'SepalWidth' else p_len if st.session_state['scatter_y'] == 'PetalLength' else p_wid
                    ax_pred.scatter(xval, yval, color=ACCENT, s=100, marker='X')
                    ax_pred.set_title('Point prédit (marqué en X)')
                    st.pyplot(fig_pred)
                except Exception:
                    pass

elif nav == 'Données':
    st.header('Données')
    if st.checkbox('Afficher tout le tableau'):
        st.dataframe(filtered_df)
    st.markdown('### Description')
    st.write(df.describe())

elif nav == 'Paramètres':
    st.header('Paramètres')
    st.write('Les paramètres globaux (thème, filtres, modèle par défaut) se trouvent dans la barre latérale.')
    if st.button('Réinitialiser le cache et l\'état'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

elif nav == 'À propos':
    st.header('À propos')
    st.write('Application Iris Data Explorer — auteur : Daniel')
    st.write('Cette application permet d\'explorer et de prédire les espèces d\'Iris à partir de mesures simples.')

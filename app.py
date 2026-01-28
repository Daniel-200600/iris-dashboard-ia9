
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn pour le modèle de prédiction
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


# --- PARTIE 4 : MODELE DE PREDICTION ---
st.header("4. Prédiction")

# Choix du modèle
model_choice = st.selectbox("Choisis un modèle", ['RandomForest', 'LogisticRegression'])

# Hyperparamètres simples
rf_estimators = None
if model_choice == 'RandomForest':
    rf_estimators = st.slider("Nombre d'arbres (n_estimators)", min_value=10, max_value=300, value=100, step=10)


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


# Entraînement (est appelé automatiquement et mis en cache)
with st.spinner('Entraînement du modèle...'):
    trained = train_model(df, model_name=model_choice, n_estimators=rf_estimators or 100)

st.subheader("Performance du modèle")
st.write(f"Accuracy (test) : {trained['acc']:.3f}")
st.text("Rapport de classification :")
st.text(trained['report'])

fig_cm, ax_cm = plt.subplots()
sns.heatmap(trained['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
            xticklabels=trained['le'].classes_, yticklabels=trained['le'].classes_)
ax_cm.set_xlabel('Prédit')
ax_cm.set_ylabel('Vrai')
ax_cm.set_title('Matrice de confusion')
st.pyplot(fig_cm)


# Interface de prédiction utilisateur
st.subheader("Prédire l'espèce d'une fleur")
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

if st.button('Prédire'):
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
            ax_pred.scatter(s_len, p_len if st.session_state['scatter_y'] == 'PetalLength' else s_wid,
                            color='black', s=100, marker='X')
            ax_pred.set_title('Point prédit (marqué en X noir)')
            st.pyplot(fig_pred)
        except Exception:
            # Ne pas planter l'app si le tracé échoue
            pass

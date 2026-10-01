# Iris Dashboard

An interactive Streamlit dashboard built on the classic **Iris** dataset (150 flowers,
3 species): explore the data, visualise it and predict the species from measurements.

## Features

- **Filters** by species and a selectable page design.
- **Visualisation**: scatter plot, boxplot, histogram and pairplot with chosen axes.
- **Model**: Random Forest (adjustable `n_estimators`) or Logistic Regression,
  trained on the fly with train/test split, accuracy, classification report and confusion matrix.
- **Prediction**: sliders for sepal/petal length and width, initialised at the dataset medians.

## Dataset

`Iris.csv` (semicolon-separated): `SepalLength`, `SepalWidth`, `PetalLength`,
`PetalWidth`, `Species`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python, Streamlit, pandas, matplotlib, seaborn, scikit-learn.

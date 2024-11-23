# %%
import importlib
import external_data.example_estimator
importlib.reload(external_data.example_estimator)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import optuna
import numpy as np
import optuna.visualization as vis

# %%
data = pd.read_parquet(Path("data") / "train.parquet")

#we add some weather data from the provided external_data dataset to the original dataset 
data =external_data.example_estimator._merge_external_data(data) 

#we reorganize the temporal data and add a season column to distinguish seasonal variations.
data = external_data.example_estimator._encode_dates(data)
data['season'] = data['month'].apply(lambda x: 1 if x in [12, 1, 2] else 
                                                 2 if x in [3, 4, 5] else
                                                 3 if x in [6, 7, 8] else 4)

print(data.head(20))

print("Information about the dataset:")
print(data.info())

print("\nVerification of data types and missing values:")
print(data.isnull().sum())
print(data.describe())

print(
    f"\nThere are {data.shape[0]} observations and {data.shape[1]} features."
    "There are no missing values. The target variable is log_count_bike."
)

# %%
## Exploratory Data Analysis 

#import ydata_profiling 
#data.profile_report()

#target distribution
plt.figure(figsize=(9,6))
sns.histplot(data['log_bike_count'], bins=50, color='purple')
plt.title('Distribution of log_bike_count')
plt.show()
print("We observe that the target is not evenly distributed but has two peaks around the value 0 and 4.")

#study of the correlation between variables
numercic_cols = data.select_dtypes(include=[np.number]).columns
corr_matrix = data[numercic_cols].corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation matrix')
plt.show()
print(
    "\nWe observe that the target is mainly correlated with the variables hour, season and latitude"
      )


#temporal analysis of log_bike_count
plt.figure(figsize=(12, 12))
plt.subplot(2,1,1)
sns.boxplot(x='hour', y='log_bike_count', data=data)
plt.title("Distribution of log_bike_count by hour of the day")
plt.subplot(2,1,2)
sns.boxplot(x='season', y='log_bike_count', data=data)
plt.title("Distribution of log_bike_count by season")
plt.show()

# %%
## Feature engineering

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), ['latitude', 'longitude', 'dd', 'ff', 'u']), 
        ('categoric', OneHotEncoder(drop='first'), ['counter_id', 'site_id'])
    ]
)

# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler

# Séparer les variables cibles et explicatives
X = data.drop(columns=['log_bike_count'])
y = data['log_bike_count']

# Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    model_choice = trial.suggest_categorical('model', ['LinearRegression', 'RandomForest', 'GradientBoosting'])

    if model_choice == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 200),  # Réduit le nombre d'estimateurs
            max_depth=trial.suggest_int('max_depth', 8, 12),  # Plage plus large pour max_depth
            random_state=42,
            n_jobs=-1
        )
    elif model_choice == 'GradientBoosting':
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 200),  # Plage pour n_estimators
            learning_rate=trial.suggest_float('learning_rate',0.5,1.5 , log=True),  # Ajustement de la plage de learning_rate
            random_state=42
        )
    else:
        model = LinearRegression()

    # Validation croisée pour évaluer les modèles
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    # Utilisation de la validation croisée pour éviter l'overfitting
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')  # 5-fold CV
    rmse = -cv_scores.mean()  # Conversion en valeur positive

    return rmse

# Créer l'étude Optuna avec un nombre réduit d'essais
study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=20)  # Augmentation du nombre d'essais

print("Best hyperparameters: ", study.best_params)

# Sélectionner le modèle final avec les meilleurs hyperparamètres
best_model_choice = study.best_params['model']
print(best_model_choice)

# Recréer le meilleur modèle en fonction des hyperparamètres optimisés
if best_model_choice == 'RandomForest':
    best_model = RandomForestRegressor(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        random_state=42,
        n_jobs=-1
    )
elif best_model_choice == 'GradientBoosting':
    best_model = GradientBoostingRegressor(
        n_estimators=study.best_params['n_estimators'],
        learning_rate=study.best_params['learning_rate'],
        random_state=42
    )
else:
    best_model = LinearRegression()

# Créer un pipeline final avec le meilleur modèle
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])

# Entraîner le modèle sur les données d'entraînement complètes
final_pipeline.fit(X_train, y_train)

# Faire une prédiction sur les données de test
final_predictions = final_pipeline.predict(X_test)

# Calculer le RMSE final
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"Final RMSE with the best model ({best_model_choice}): {final_rmse}")

# Exemple de prédiction sur de nouvelles données (si disponibles)
# new_data = pd.DataFrame(...)  # Remplissez avec vos nouvelles données
# predictions = final_pipeline.predict(new_data)


# %%
data_final_test = pd.read_parquet("/Users/gauthiercaceres/bike_counters/data/final_test.parquet")
data_final_test =external_data.example_estimator._merge_external_data(data_final_test)
predictions = final_pipeline.predict(data_final_test)
results_df = pd.DataFrame({
    'id': data_final_test['counter_id'],
    'log_bike_count': predictions
})
results_df.to_csv('predictions.csv', index=False)

# %%
import joblib

# Sauvegarder le modèle dans un fichier
joblib.dump(final_pipeline, 'final_pipeline_model_23_11.pkl')


# %%




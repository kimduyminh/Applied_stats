from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#read data
yes_df = pd.read_csv('clean_data/vpn120s_cleaned.csv')
no_df = pd.read_csv('clean_data/novpn120s_cleaned.csv')


# Create labels for dataset
yes_df['label'] = 1  
no_df['label'] = 0   

#combine 2 dataset
combined_df = pd.concat([yes_df, no_df], ignore_index=True)

# separate features and specify target variable
X = combined_df.drop(columns=['label'])
y = combined_df['label']

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cpu_cores=12
# Define the parameter grid for the Random Forest
param_grid_rf = {
    'rfr__n_estimators': [100, 500, 1000],
    'rfr__max_depth': [None, 10, 20],
    'rfr__min_samples_split': [2, 5, 10]
}

# Define the parameter grid for the KNN
param_grid_knn = {
    'knn__n_neighbors': [1,2,3,4, 5,6, 7]
}

# Add a new base estimator
estimators = [
    ('rfr', RandomForestClassifier(random_state=42, n_jobs=cpu_cores)),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True))
]

# Create the stacking classifier
reg = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegressionCV(n_jobs=cpu_cores, solver='liblinear')
)

# Combine the parameter grids
param_grid = {**param_grid_rf, **param_grid_knn}

# Perform Grid Search
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, n_jobs=cpu_cores, scoring='accuracy',verbose=10)
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

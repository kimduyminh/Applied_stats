import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

#read data
yes_df = pd.read_csv('clean_data/vpn15s_cleaned.csv')
no_df = pd.read_csv('clean_data/novpn15s_cleaned.csv')


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

#specifying the category and numerical columns
cate_cols=[cname for cname in X_train.columns if X_train[cname].dtype=="object"]
nume_cols=[cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']]

# preprocessing to deal with different type of datas
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, nume_cols),
        ('cat', categorical_transformer, cate_cols)
    ])


# train the model
knn = Pipeline(steps=[('preprocessor', preprocessor),
                    ('model', DecisionTreeClassifier(max_leaf_nodes=1000))
                             ])
knn.fit(X_train, y_train)

# predicting some value
y_pred = knn.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# print the report
print(classification_report(y_test, y_pred))

# view some first prediction
print("First few predictions:", y_pred[:5])
print("First few actual values:", y_test[:5].values)
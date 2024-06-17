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

cpu_cores=8

def get_accuracy_score(n,traX,reaX,traY,reaY):
    model_2=DecisionTreeClassifier(max_depth=n,random_state=1)
    pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model_2)
                                ])
    pipeline2.fit(traX,traY)
    predict_val=pipeline2.predict(reaX)
    accuracy_test=accuracy_score(reaY,predict_val)
    return accuracy_test

#finding best tree number to maximize accuracy
n_list=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,50,100,200,500,1000,2000,5000,10000]
accuracy_data=[]
for i in n_list:
    print("\n")
    print("Trying tree numbers: "+str(i))
    a=get_accuracy_score(i,X_train,X_test,y_train,y_test)
    accuracy_data.append(a)
    print("Accuracy: "+str(a))
best_number_of_tree = n_list[accuracy_data.index(max(accuracy_data))]
print("BEST NUMBER OF TREES")
print(best_number_of_tree)

dt=Pipeline(steps=[('preprocessor', preprocessor),
                    ('model', DecisionTreeClassifier(max_depth=best_number_of_tree))
                             ])
dt.fit(X_train,y_train)

# predicting some value
y_pred = dt.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# print the report
print(classification_report(y_test, y_pred))

# view some first prediction
print("First few predictions:", y_pred[:5])
print("First few actual values:", y_test[:5].values)
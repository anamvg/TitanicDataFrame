import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv("train.csv")

# 2. Basic preprocessing (handle missing values)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 3. Define features (X) and target (y)
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# 4. Split into training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Save trained model to disk
joblib.dump(clf, "model.joblib")
print("Model saved successfully as model.joblib!")
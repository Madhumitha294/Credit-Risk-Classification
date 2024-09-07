import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = 'C:/Users/madhu/OneDrive/Desktop/credit_customers(DS).csv'
data = pd.read_csv(file_path)

# Print dataset information
print("Dataset Info:")
print(data.info())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Fill missing values if necessary
data.ffill(inplace=True)  # Forward fill missing values

# Check for and drop duplicate records
data.drop_duplicates(inplace=True)

# Target variable is 'class'
target_variable = 'class'
if target_variable not in data.columns:
    raise ValueError(f"Target variable '{target_variable}' not found in the dataset.")

# Print unique values in target variable
print("Unique values in the 'class' column:")
print(data[target_variable].unique())
print("Value counts in the 'class' column:")
print(data[target_variable].value_counts())

# Ensure the dataset has at least two classes for training
if len(data[target_variable].unique()) < 2:
    raise ValueError("Dataset must contain at least two classes for training")

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target variable
X = data.drop(target_variable, axis=1)
y = data[target_variable]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle imbalanced data
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=20000),
    'KNN Classifier': KNeighborsClassifier(),
    'SVM Linear': SVC(kernel='linear', probability=True),
    'SVM RBF': SVC(kernel='rbf', probability=True)
}

# Track the best model
best_model = None
best_accuracy = 0

# Train and evaluate models
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {accuracy:.2f}")
        print(f"{name} Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
        print(f"{name} Classification Report:\n {classification_report(y_test, y_pred)}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name
    except Exception as e:
        print(f"Error training {name}: {e}")

print(f"Best model is {best_model} with accuracy of {best_accuracy:.2f}")

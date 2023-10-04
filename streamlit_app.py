import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification  # Use a placeholder dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np  # Import numpy for min and max functions


# Create a placeholder dataset (you should replace this with real data)
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
df = pd.DataFrame(data=X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
df['target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42
)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a Streamlit web app
st.title("Heat Wave Classifier")

# Sidebar for user input
st.sidebar.header("User Input")

# Display dataset information
st.sidebar.subheader("Dataset Info")
st.sidebar.write(f"Number of samples: {len(df)}")
st.sidebar.write(f"Number of features: {len(df.columns) - 1}")

# Display classifier information
st.sidebar.subheader("Classifier Info")
st.sidebar.write(f"Classifier used: Random Forest")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")

# Display the dataset
st.subheader("Heat Waves Dataset")
st.write(df)

# Display a heatmap of the dataset
st.subheader("Heatmap of the Dataset")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot()

# Additional EDA components

# Descriptive statistics of the dataset
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Box plots for numeric features
st.subheader("Box Plots")
numeric_features = df.select_dtypes(include=['number']).columns
selected_feature = st.selectbox("Select a feature:", numeric_features)
if selected_feature:
    st.write(sns.boxplot(x=df[selected_feature]))
    st.pyplot()

# Pairwise scatter plot for numeric features
st.subheader("Pairwise Scatter Plot")
sns.pairplot(df, hue='target')
st.pyplot()

# Distribution of target classes
st.subheader("Distribution of Target Classes")
st.write(df['target'].value_counts())

# Countplot for target classes
st.subheader("Countplot for Target Classes")
sns.countplot(data=df, x='target')
st.pyplot()

# Correlation matrix
st.subheader("Correlation Matrix")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot()

# Distribution of target classes
st.subheader("Distribution of Target Classes")
st.write(df['target'].value_counts())

# Countplot for target classes
st.subheader("Countplot for Target Classes")
sns.countplot(data=df, x='target')
st.pyplot()

# Correlation matrix
st.subheader("Correlation Matrix")
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot()

# User input for prediction
st.subheader("Make Predictions")
st.write("Enter values for features to classify a heat wave:")
user_input = {}
for feature_name in df.columns[:-1]:  # Exclude the 'target' column
    user_input[feature_name] = st.sidebar.number_input(
        f"Enter {feature_name}", min_value=np.min(df[feature_name]), max_value=np.max(df[feature_name])
    )

if st.button("Classify Heat Wave"):
    user_input_df = pd.DataFrame(user_input, index=[0])
    prediction = rf_classifier.predict(user_input_df)
    prediction_proba = rf_classifier.predict_proba(user_input_df)

    st.subheader("Prediction")
    st.write(f"Predicted Heat Wave Type: {prediction[0]}")
    st.write("Prediction Probabilities:")
    st.write(prediction_proba[0])

# Display classification report
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# Display accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Display the feature importances
st.subheader("Feature Importances")
feature_importances = pd.DataFrame(
    {'Feature': df.columns[:-1], 'Importance': rf_classifier.feature_importances_}
)
st.bar_chart(feature_importances.set_index('Feature').sort_values(by='Importance', ascending=False))

# Display the most important feature
st.subheader("Most Important Feature")
most_important_feature = feature_importances.loc[feature_importances['Importance'].idxmax()]['Feature']
st.write(f"The most important feature is: {most_important_feature}")

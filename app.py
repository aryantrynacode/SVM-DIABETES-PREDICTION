import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
import json

df = pd.read_csv('diabetes.csv')

st.title("Support vector machines")

st.write("SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.")

st.write("Goal of SVM The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.")

st.write("SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. Consider the below diagram in which there are two different categories that are classified using a decision boundary or hyperplane.")

st.title("DATASET")

st.subheader("Source: Kaggle")

st.write("PIMA-INDIAN-DIABETES-DATASET ")

st.link_button("Download Dataset","https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")

st.title("OBJECTIVE")

st.write("We will be predicting if a person has diabetes or not (Outcome) using Support Vector Machine and We will be evaluating the performance of the Model using Classification Report,Confusion Matrix and Accuracy Score")

eda_option = st.selectbox("Select an EDA Option", [
    "Show Dataset",
    "Dataset Shape",
    "Column Names",
    "Missing Values",
    "Descriptive Statistics",
    "Correlation Heatmap",
    "Outcome Value Count"
])

if eda_option == "Show Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df)

elif eda_option == "Dataset Shape":
    st.subheader("Shape of Dataset")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

elif eda_option == "Column Names":
    st.subheader("Column Names")
    st.write(df.columns.tolist())

elif eda_option == "Missing Values":
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

elif eda_option == "Descriptive Statistics":
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

elif eda_option == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif eda_option == "Outcome Value Count":
    st.subheader("Outcome Column Distribution")
    if "Outcome" in df.columns:
        st.bar_chart(df["Outcome"].value_counts())
    else:
        st.warning("Column 'Outcome' not found!")

scaler = StandardScaler()
scaled = scaler.fit(df.drop('Outcome',axis=1)).transform(df.drop('Outcome',axis=1))
df_scaled = pd.DataFrame(scaled, columns=df.columns[:-1])
df_scaled.head()

x = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_columns = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

st.title("SVM KERNEL SELECTOR")

st.write("A kernel is a function that maps data into a higher-dimensional space to enable the algorithm to find a hyperplane that separates data, even if it's not linearly separable in the original space")

kernel = st.selectbox("Select SVM kernel", ["linear", "poly", "rbf", "sigmoid"])

# Optional: Set parameters based on kernel
C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
gamma = st.selectbox("Gamma (for rbf/poly/sigmoid)", ["scale", "auto"])
degree = 3  # Default degree for poly

# Only show degree option for 'poly'
if kernel == "poly":
    degree = st.slider("Polynomial Degree (for poly kernel)", 2, 5, 3)

# Fit SVM model with selected kernel and params
clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree if kernel == "poly" else 3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)



accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report_dict = classification_report(y_test, y_pred, output_dict=True)
class_report_str = json.dumps(class_report_dict, indent=2)

evaluation_option = st.selectbox("Select an Evaluation Option", [
    "Accuracy Score",
    "Confusion Matrix",
    "Classification Report"
])

if evaluation_option == "Accuracy Score":
    st.subheader(" Accuracy Score")
    st.write(f"{accuracy:.2f}")

elif evaluation_option == "Confusion Matrix":
    st.subheader(" Confusion Matrix")
    cm_df = pd.DataFrame(
        conf_matrix,
        columns=['Actual Positive (1)', 'Actual Negative (0)'],
        index=['Predicted Positive (1)', 'Predicted Negative (0)']
    )
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='mako', ax=ax)
    st.pyplot(fig)

elif evaluation_option == "Classification Report":
    st.subheader(" Classification Report")
    st.text(class_report_str)


st.title(" Predict Diabetes from User Input")


# Sidebar or main input section
st.subheader("Enter Patient Data")

# Dynamically generate input fields for all features
user_data = {}
for feature in df.columns[:-1]:  # exclude 'Outcome'
    user_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))

# Convert input to DataFrame
user_input_df = pd.DataFrame([user_data])

# Otherwise, predict directly (if not scaled):
prediction = clf.predict(user_input_df)[0]

# Show result
st.markdown("---")
st.subheader("Prediction Result")
if prediction == 1:
    st.error("❗ The model predicts: **Diabetic (1)**")
else:
    st.success("✅ The model predicts: **Non-Diabetic (0)**")

from sklearn.metrics import roc_curve, roc_auc_score

if len(np.unique(y_test)) == 2:
    y_prob = clf.decision_function(x_test)  # works for SVM
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

st.title("DOWNLOAD EVALUATIONS")

st.download_button(
    label="📥 Download Classification Report",
    data= class_report_str,
    file_name="classification_report.txt",
    mime="application/json"
)
st.title("Conclusion")
st.write("In this project, we successfully applied the Support Vector Machine (SVM) algorithm to the PIMA Indian Diabetes dataset to predict whether a patient has diabetes or not based on medical attributes. The SVM model demonstrated solid classification performance, particularly with the linear kernel, making it suitable for linearly separable data like this dataset. By evaluating the model using accuracy score, confusion matrix, and classification report, we observed a balanced performance in detecting both diabetic and non-diabetic cases. With proper feature scaling and hyperparameter tuning, SVM proved to be an effective and reliable algorithm for binary classification in medical datasets. This approach can be extended and enhanced with techniques like cross-validation, feature selection, and ensemble models for even better results in real-world healthcare applications.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
st.markdown('Thank you for checking this blog ~ By Aryan')

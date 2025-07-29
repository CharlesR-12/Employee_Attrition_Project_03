#=============================================================================
# 👨‍💼 Employee Attrition Prediction Project
#=============================================================================

# This script contains the complete pipeline for the Employee Attrition Prediction project.
          # It is structured into two main parts:
            #   PART 1: Model Training and Preprocessing (typically run once to generate .pkl files)
            #   PART 2: Streamlit Dashboard Application (the interactive web application)
#=============================================================================
# Note: The 'file_path' for the CSV dataset should be updated to your local path.
file_path = r"E:\Guvi_Class\.venv\Mini_Projects_Data\Project_03_Employee_Attrition\Employee-Attrition - Employee-Attrition.csv"
#=============================================================================

# -----------------------------------------
# Global Imports Function
# -----------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle # For saving/loading models and scalers


# =============================================================================
# ✨ PART 1: MODEL TRAINING AND PREPROCESSING (Typically run once to generate .pkl files)
#=============================================================================

# This block executes only when the script is run directly (e.g., `python your_script_name.py`).
# It handles data loading, preprocessing, feature engineering, model training, and saving the trained model and scaler for use in the Streamlit application.
if __name__ == "__main__":
    # -----------------------------------------
    # Stage 1: Data Collection & Initial Preprocessing
    # -----------------------------------------

    # Step 1.1: Load CSV Dataset
    # Loads the employee attrition dataset into a pandas DataFrame.
    df = pd.read_csv(file_path)
    # -----------------------------------------

    # Step 1.2: Display Basic Dataset Information
    # Provides an initial overview of the dataset's structure, data types, and missing values.
    print("--- Dataset Information ---")
    print("Shape of Dataset:", df.shape)
    print("\nFirst 5 Rows:\n", df.head())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    # -----------------------------------------

    # Step 1.3: Drop Unnecessary Constant Columns
    # Identifies and removes columns with only one unique value, as they provide no predictive power.
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print("\nDropping constant columns:", constant_cols)
        df.drop(columns=constant_cols, inplace=True)
    # -----------------------------------------

    # Step 1.4: Remove Irrelevant Identifier Columns
    # Removes 'EmployeeNumber' as it's typically a unique identifier and not a predictive feature.
    if 'EmployeeNumber' in df.columns:
        print("\nDropping 'EmployeeNumber' column.")
        df.drop('EmployeeNumber', axis=1, inplace=True)
    # -----------------------------------------

    print("\n--- Stage 1: Initial Preprocessing Completed ---")
    print("Cleaned Dataset Shape:", df.shape)
    print("Columns After Initial Cleanup:\n", df.columns.tolist())


    # -----------------------------------------
    # Stage 2: Data Preprocessing & Encoding
    # -----------------------------------------

    # Step 2.1: Encode Binary Target Column ('Attrition': Yes → 1, No → 0)
    # Converts the categorical target variable 'Attrition' into a numerical format.
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    # -----------------------------------------

    # Step 2.2: Identify Categorical Columns for Encoding
    # Separates object-type columns into binary and multi-category lists for appropriate encoding.
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\nCategorical Columns identified for encoding:\n", cat_cols)
    # -----------------------------------------

    # Step 2.3: Perform Label Encoding and One-Hot Encoding
    # Applies Label Encoding to binary categorical features (e.g., Gender, OverTime) to multi-category nominal features (e.g., Department, JobRole) using `drop_first=True` to prevent multicollinearity.
    label_encoder = LabelEncoder()
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    multi_cols = [col for col in cat_cols if df[col].nunique() > 2]

    for col in binary_cols:
        df[col] = label_encoder.fit_transform(df[col])

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    # -----------------------------------------

    # Step 2.4: Handle Outliers (Z-score based clipping)
    # Filters out rows where any numerical feature has an absolute Z-score greater than 3, treating these as extreme outliers.
    from scipy.stats import zscore
    numerical_df_for_zscore = df.select_dtypes(include=np.number).drop(columns=['Attrition'], errors='ignore')
    z_scores = np.abs(zscore(numerical_df_for_zscore))
    df = df[(z_scores < 3).all(axis=1)]
    # -----------------------------------------

    print("\n--- Stage 2: Data Preprocessing & Encoding Completed ---")
    print("New Shape After Encoding & Outlier Handling:", df.shape)
    print("Dataset Columns After Encoding:\n", df.columns.tolist())


    # -----------------------------------------
    # Stage 3: Exploratory Data Analysis (EDA)
    # -----------------------------------------

    # Configures matplotlib for consistent plot sizing.
    plt.rcParams["figure.figsize"] = (5, 4)

    # Step 3.1: Attrition Distribution - Visualizes the count of employees who stayed (0) vs. left (1).
    plt.figure(figsize=(5, 4))
    sns.countplot(x='Attrition', data=df)
    plt.title("Attrition Count (0 = Stayed, 1 = Left)", fontweight='bold', color='blue')
    plt.show()
    # -----------------------------------------

    # Step 3.2: Attrition Rate by Gender - Compares attrition rates between genders.
    plt.figure(figsize=(5, 4))
    sns.countplot(x='Gender', hue='Attrition', data=df)
    plt.title("Attrition by Gender", fontweight='bold', color='blue')
    plt.show()
    # -----------------------------------------

    # Step 3.3: Attrition by Job Role - Visualizes attrition across different job roles using the original JobRole names.
    original_df_for_plot = pd.read_csv(file_path)
    original_df_for_plot['Attrition'] = original_df_for_plot['Attrition'].map({'Yes': 1, 'No': 0})
    plt.figure(figsize=(5, 4))
    sns.countplot(y='JobRole', hue='Attrition', data=original_df_for_plot, order=original_df_for_plot['JobRole'].value_counts().index, palette='viridis')
    plt.title("Attrition by Job Role", fontweight='bold', color='blue')
    plt.xlabel("Number of Employees")
    plt.ylabel("Job Role")
    plt.show()
    # -----------------------------------------

    # Step 3.4: Monthly Income Distribution - Shows the distribution of monthly income.
    plt.figure(figsize=(5, 4))
    sns.histplot(df['MonthlyIncome'], kde=True, bins=30)
    plt.title("Monthly Income Distribution", fontweight='bold', color='blue')
    plt.xlabel("Monthly Income")
    plt.ylabel("Count")
    plt.show()
    # -----------------------------------------

    # Step 3.5: Boxplot – Monthly Income vs. Attrition - Analyzes the relationship between monthly income and attrition.
    plt.figure(figsize=(5, 4))
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
    plt.title("Monthly Income vs Attrition", fontweight='bold', color='blue')
    plt.xlabel("Attrition (0=Stayed, 1=Left)")
    plt.ylabel("Monthly Income")
    plt.show()
    # -----------------------------------------

    # Step 3.6: Attrition vs. WorkLifeBalance - Examines attrition based on work-life balance ratings.
    if 'WorkLifeBalance' in df.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x='WorkLifeBalance', hue='Attrition', data=df)
        plt.title("Attrition by Work-Life Balance Rating", fontweight='bold', color='blue')
        plt.xlabel("Work-Life Balance Rating")
        plt.ylabel("Number of Employees")
        plt.show()
    # -----------------------------------------

    # Step 3.7: Attrition by Overtime - Investigates the impact of overtime on attrition.
    if 'OverTime' in df.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x='OverTime', hue='Attrition', data=df)
        plt.title("Attrition by Overtime", fontweight='bold', color='blue')
        plt.xlabel("Overtime (0=No, 1=Yes)")
        plt.ylabel("Number of Employees")
        plt.show()
    # -----------------------------------------

    # Step 3.8: Correlation Heatmap (Numerical Features) - Visualizes the correlation matrix of numerical features.
    plt.figure(figsize=(5, 4))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False, fmt=".2f")
    plt.title("Correlation Heatmap", fontweight='bold', color='blue')
    plt.show()
    # -----------------------------------------

    print("\n--- Stage 3: EDA Visualization Completed ---")

    # -----------------------------------------
    # Stage 4: Feature Engineering
    # -----------------------------------------

    # Step 4.1: Create 'TenureCategory' based on YearsAtCompany - Categorizes employees into tenure groups based on their years at the company.
    def tenure_group(years):
        if years <= 2:
            return '0-2 Years'
        elif years <= 5:
            return '3-5 Years'
        elif years <= 10:
            return '6-10 Years'
        else:
            return '10+ Years'
    df['TenureCategory'] = df['YearsAtCompany'].apply(tenure_group)
    # -----------------------------------------

    # Step 4.2: Create 'EngagementScore' - Creates a composite score by averaging multiple satisfaction metrics.
    if set(['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']).issubset(df.columns):
        df['EngagementScore'] = df[['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    # -----------------------------------------

    # Step 4.3: Create 'ExperienceLevel' from TotalWorkingYears
    # Categorizes employees based on their total working years.
    def experience_group(years):
        if years < 3:
            return 'Fresher'
        elif years <= 7:
            return 'Junior'
        elif years <= 15:
            return 'Mid-Level'
        else:
            return 'Senior'
    df['ExperienceLevel'] = df['TotalWorkingYears'].apply(experience_group)
    # -----------------------------------------

    # Step 4.4: Create 'PromotionGapFlag' - Flags employees who have not received a promotion in 5 or more years.
    df['PromotionGapFlag'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x >= 5 else 0)
    # -----------------------------------------

    # Step 4.5: Encode newly created categorical features - Applies one-hot encoding to 'TenureCategory' and 'ExperienceLevel'.
    df = pd.get_dummies(df, columns=['TenureCategory', 'ExperienceLevel'], drop_first=True)
    # -----------------------------------------

    print("\n--- Stage 4: Feature Engineering Completed ---")
    print("New Columns Added:\n", ['EngagementScore', 'PromotionGapFlag'] + [col for col in df.columns if 'TenureCategory_' in col or 'ExperienceLevel_' in col])

    # -----------------------------------------
    # Stage 5: Model Building & Training
    # -----------------------------------------

    # Step 5.1: Define Target and Features - Separates the target variable ('Attrition') from the feature set (X).
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    # -----------------------------------------

    # Step 5.2: Split Dataset into Training and Testing Sets - Divides the dataset into training (80%) and testing (20%) sets, using stratification to maintain the class distribution of 'Attrition'.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # -----------------------------------------

    # Step 5.3: Feature Scaling (Standardization) - Scales numerical features using StandardScaler, which is recommended for Logistic Regression.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # -----------------------------------------

    # Step 5.4: Train Logistic Regression Model - Initializes and trains a Logistic Regression model.
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    # -----------------------------------------

    # Step 5.5: Train Random Forest Classifier - Initializes and trains a Random Forest Classifier.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) # Random Forest is less sensitive to scaling
    y_pred_rf = rf_model.predict(X_test)
    # -----------------------------------------

    # Step 5.6: Define Model Evaluation Function - A helper function to print common classification evaluation metrics.
    def evaluate_model(y_true, y_pred, model_name):
        print(f"\n--- Evaluation: {model_name} ---")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print("Classification Report:\n", classification_report(y_true, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_true, y_pred))
    # -----------------------------------------

    # Step 5.7: Evaluate Trained Models - Calls the evaluation function for both Logistic Regression and Random Forest models.
    evaluate_model(y_test, y_pred_lr, "Logistic Regression")
    evaluate_model(y_test, y_pred_rf, "Random Forest")
    # -----------------------------------------

    # Step 5.8: Multi-model comparison with hyperparameter tuning (GridSearchCV Example)
    print("\n--- Step 5.8: Hyperparameter Tuning for Random Forest (Example) ---")
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=3, # 3-fold cross-validation
                               scoring='roc_auc', # Optimize for ROC-AUC
                               n_jobs=-1, # Use all available cores
                               verbose=1) # Print progress
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")

    # Use the best estimator from GridSearchCV
    tuned_rf_model = grid_search.best_estimator_
    y_pred_tuned_rf = tuned_rf_model.predict(X_test)
    evaluate_model(y_test, y_pred_tuned_rf, "Tuned Random Forest")
    # -----------------------------------------

    print("\n--- Stage 5: Model Training Completed ---")

    # Step 5.9: Save Trained Model and Scaler - Saves the trained Random Forest model and the fitted StandardScaler using pickle. 
    # For this example, we'll save the *tuned* Random Forest model.
    pickle.dump(tuned_rf_model, open("random_forest_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    # -----------------------------------------
    print("\n--- Trained model and scaler saved as 'random_forest_model.pkl' and 'scaler.pkl'. ---")


# =============================================================================
# ✨ PART 2: STREAMLIT DASHBOARD APPLICATION (app.py equivalent)
# =============================================================================

# -----------------------------------------
# Stage 6: Streamlit Dashboard App Setup
# -----------------------------------------

# Step 6.1: Streamlit Page Configuration - Sets the page title and layout for the Streamlit application.
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
# -----------------------------------------

# Step 6.2: Custom CSS for Enhanced Aesthetics - Injects custom CSS to control text size, color, and centering for better visual presentation.
st.markdown("""
<style>
/* Center all content within the main block container */
div.block-container {
    text-align: center;
    padding-top: 2rem; /* Add some padding at the top */
    padding-bottom: 2rem; /* Add some padding at the bottom */
    padding-left: 0rem; /* Removed left padding for full width */
    padding-right: 0rem; /* Removed right padding for full width */
}

/* Ensure headers are centered and maintain color/size */
h1 { /* Specific styling for h1 (st.title) */
    font-size: 5.5em; /* Significantly larger title */
    font-weight: bold;
    color: red; /* Changed to red as requested */
    text-align: center;
    width: 100%;
    margin-bottom: 0.5em; /* Space below title */
}

h2 {
    font-size: 1.8em; /* Adjust other header sizes */
    color: blue !important; /* Changed to blue, added !important for override */
    font-weight: bold; /* Changed to bold */
    text-align: center;
    width: 100%;
    margin-top: 1.5em; /* Space above subheaders */
    margin-bottom: 0.8em;
}

h3, h4, h5, h6 {
    font-size: 1.4em;
    color: red; /* Changed to red */
    font-weight: bold; /* Changed to bold */
    text-align: center;
    width: 100%;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

/* Targeting text elements within st.write and st.markdown for size/color */
.st-emotion-cache-10qj07j, .st-emotion-cache-10qj07j p { /* This class often applies to st.markdown, st.write, st.text etc. */
    font-size: 1.1em;
    color: red; /* Changed to red */
    font-weight: bold; /* Changed to bold */
    text-align: center;
}

/* For the prediction result text (st.error/st.success), ensure it's centered */
.stAlert {
    text-align: center;
    font-size: 1.3em;
    font-weight: bold;
    color: red !important; /* Ensure red for alerts */
}

/* For the input features section in the sidebar, align its header to left for better readability */
[data-testid="stSidebar"] .st-emotion-cache-10qj07j h2 {
    text-align: center; /* Changed to center */
    color: black; /* Changed to black */
    font-weight: bold;
}

/* General style for all sidebar input labels (sliders, etc.) */
[data-testid="stSidebar"] label {
    text-align: left;
    display: block; /* Ensures label takes full width for text-align to work */
    color: black; /* Default to black for general labels */
    font-weight: bold;
    font-size: 1em;
}

/* Specific style for st.sidebar.selectbox labels */
/* This targets the label element directly associated with the selectbox widget */
[data-testid="stSidebar"] .stSelectbox label {
    color: blue !important; /* Blue color for selectbox labels, !important to override */
}

/* For the sidebar input widgets themselves, ensure they are left-aligned if possible */
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stSelectbox {
    text-align: left;
}

/* Style for the horizontal rule to make it a box */
hr {
    border: 2px solid #BBBBBB; /* Full border */
    margin-top: 2em;
    margin-bottom: 2em;
    padding: 0.5em; /* Add some padding inside the box */
    border-radius: 8px; /* Rounded corners for the box */
    background-color: #f0f2f6; /* Light background for the box */
}

/* For the expander header text */
[data-testid="stExpander"] summary > div {
    text-align: center;
    width: 100%; /* Ensure the container itself takes full width */
}

/* Style for the selectbox label in the main content area */
/* This targets labels that are children of Streamlit's block containers, but not in the sidebar */
div[data-testid^="stVerticalBlock"] label,
div[data-testid^="stHorizontalBlock"] label,
div[data-testid="stForm"] label {
    font-size: 1.5em; /* Increased size */
    font-weight: bold; /* Bold text */
    color: blue !important; /* Blue color, !important to override defaults */
    text-align: center; /* Center the label text */
    width: 100%; /* Ensure it takes full width for centering */
    display: block; /* Important for text-align to work on label */
}

/* Also ensure the selectbox itself is centered within its column */
.stSelectbox {
    margin-left: auto;
    margin-right: auto;
    width: fit-content; /* Or a specific width if desired, but fit-content is good for centering */
}
</style>
""", unsafe_allow_html=True)
# -----------------------------------------

# Step 6.3: Load Trained Model and Scaler, Loads the pre-trained Random Forest model and StandardScaler from their .pkl files.
# Includes error handling for FileNotFoundError.
try:
    model = pickle.load(open("random_forest_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Halts the app if essential files are missing.

# Helper functions for feature engineering (replicated for Streamlit input processing)
def tenure_group_input(years):
    if years <= 2: return '0-2 Years'
    elif years <= 5: return '3-5 Years'
    elif years <= 10: return '6-10 Years'
    else: return '10+ Years'

def experience_group_input(years):
    if years < 3: return 'Fresher'
    elif years <= 7: return 'Junior'
    elif years <= 15: return 'Mid-Level'
    else: return 'Senior'
# -----------------------------------------

# Step 6.4: Dynamically Determine Expected Features for Consistency - This block is crucial for ensuring that the input features from the Streamlit sidebar
# match the exact order and structure of features the model was trained on. - It simulates the preprocessing steps on a dummy DataFrame to get the expected column list.
temp_df_for_features_streamlit = pd.read_csv(file_path)
if 'EmployeeNumber' in temp_df_for_features_streamlit.columns:
    temp_df_for_features_streamlit.drop('EmployeeNumber', axis=1, inplace=True)
constant_cols_temp_streamlit = [col for col in temp_df_for_features_streamlit.columns if temp_df_for_features_streamlit[col].nunique() == 1]
temp_df_for_features_streamlit.drop(columns=constant_cols_temp_streamlit, inplace=True)
temp_df_for_features_streamlit['Attrition'] = temp_df_for_features_streamlit['Attrition'].map({'Yes': 1, 'No': 0})
cat_cols_temp_streamlit = temp_df_for_features_streamlit.select_dtypes(include=['object']).columns.tolist()
binary_cols_temp_streamlit = [col for col in cat_cols_temp_streamlit if temp_df_for_features_streamlit[col].nunique() == 2]
multi_cols_temp_streamlit = [col for col in cat_cols_temp_streamlit if temp_df_for_features_streamlit[col].nunique() > 2]
label_encoder_temp_streamlit = LabelEncoder()
for col in binary_cols_temp_streamlit:
    temp_df_for_features_streamlit[col] = label_encoder_temp_streamlit.fit_transform(temp_df_for_features_streamlit[col])
temp_df_for_features_streamlit = pd.get_dummies(temp_df_for_features_streamlit, columns=multi_cols_temp_streamlit, drop_first=True)
temp_df_for_features_streamlit['TenureCategory'] = temp_df_for_features_streamlit['YearsAtCompany'].apply(tenure_group_input)
if set(['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']).issubset(temp_df_for_features_streamlit.columns):
    temp_df_for_features_streamlit['EngagementScore'] = temp_df_for_features_streamlit[['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
temp_df_for_features_streamlit['ExperienceLevel'] = temp_df_for_features_streamlit['TotalWorkingYears'].apply(experience_group_input)
temp_df_for_features_streamlit['PromotionGapFlag'] = temp_df_for_features_streamlit['YearsSinceLastPromotion'].apply(lambda x: 1 if x >= 5 else 0)
temp_df_for_features_streamlit = pd.get_dummies(temp_df_for_features_streamlit, columns=['TenureCategory', 'ExperienceLevel'], drop_first=True)
expected_features = temp_df_for_features_streamlit.drop('Attrition', axis=1).columns.tolist()
# -----------------------------------------

# Step 6.5: Application Title and Description for the Streamlit app.
st.title("👨‍💼👩🏻‍💼 Employee Attrition Prediction 👩🏻‍💼👨‍💼 ")
# -----------------------------------------

# Step 6.6: Sidebar for User Input Features - Defines a function to create interactive input widgets in the sidebar for employee features.
st.sidebar.header("👷🏼‍♂️👩🏻‍🔧 Employee Input Features 👩🏻‍🔧👷🏼‍♂️")

def user_input_features():
    # Numerical Inputs
    Age = st.sidebar.slider("Age", 18, 60, 30)
    DailyRate = st.sidebar.slider("Daily Rate", 100, 1500, 800)
    DistanceFromHome = st.sidebar.slider("Distance From Home (miles)", 1, 30, 10)
    HourlyRate = st.sidebar.slider("Hourly Rate", 30, 100, 65)
    JobLevel = st.sidebar.slider("Job Level", 1, 5, 1)
    MonthlyIncome = st.sidebar.slider("Monthly Income", 1000, 25000, 5000)
    MonthlyRate = st.sidebar.slider("Monthly Rate", 2000, 27000, 10000)
    NumCompaniesWorked = st.sidebar.slider("Number of Companies Worked", 0, 9, 1)
    PercentSalaryHike = st.sidebar.slider("Percent Salary Hike (%)", 11, 25, 15)
    StockOptionLevel = st.sidebar.selectbox("Stock Option Level", [0, 1, 2, 3])
    TotalWorkingYears = st.sidebar.slider("Total Working Years", 0, 40, 10)
    TrainingTimesLastYear = st.sidebar.slider("Training Times Last Year", 0, 6, 2)
    YearsAtCompany = st.sidebar.slider("Years at Company", 0, 40, 5)
    YearsInCurrentRole = st.sidebar.slider("Years in Current Role", 0, 18, 3)
    YearsSinceLastPromotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
    YearsWithCurrManager = st.sidebar.slider("Years with Current Manager", 0, 17, 3)

    # Satisfaction/Involvement Ratings (1=Low, 2=Medium, 3=High, 4=Very High)
    EnvironmentSatisfaction = st.sidebar.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    JobInvolvement = st.sidebar.selectbox("Job Involvement", [1, 2, 3, 4])
    JobSatisfaction = st.sidebar.selectbox("Job Satisfaction", [1, 2, 3, 4])
    PerformanceRating = st.sidebar.selectbox("Performance Rating", [1, 2, 3, 4])
    RelationshipSatisfaction = st.sidebar.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
    WorkLifeBalance = st.sidebar.selectbox("Work-Life Balance", [1, 2, 3, 4])
    Education = st.sidebar.selectbox("Education (1 'Below College' to 5 'Doctor')", [1, 2, 3, 4, 5])

    # Categorical Inputs
    Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    OverTime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    BusinessTravel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
    Department = st.sidebar.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
    EducationField = st.sidebar.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
    JobRole = st.sidebar.selectbox("Job Role", [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Divorced", "Married", "Single"])

    # Collect raw user inputs into a dictionary
    raw_data = {
        'Age': Age, 'DailyRate': DailyRate, 'DistanceFromHome': DistanceFromHome,
        'HourlyRate': HourlyRate, 'JobLevel': JobLevel, 'MonthlyIncome': MonthlyIncome,
        'MonthlyRate': MonthlyRate, 'NumCompaniesWorked': NumCompaniesWorked,
        'PercentSalaryHike': PercentSalaryHike, 'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears, 'TrainingTimesLastYear': TrainingTimesLastYear,
        'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion, 'YearsWithCurrManager': YearsWithCurrManager,
        'EnvironmentSatisfaction': EnvironmentSatisfaction, 'JobInvolvement': JobInvolvement,
        'JobSatisfaction': JobSatisfaction, 'PerformanceRating': PerformanceRating,
        'RelationshipSatisfaction': RelationshipSatisfaction, 'WorkLifeBalance': WorkLifeBalance,
        'Education': Education, 'Gender': Gender, 'OverTime': OverTime,
        'BusinessTravel': BusinessTravel, 'Department': Department,
        'EducationField': EducationField, 'JobRole': JobRole, 'MaritalStatus': MaritalStatus
    }

    # Convert raw data to a DataFrame
    input_df = pd.DataFrame([raw_data])

    # Apply the same preprocessing and feature engineering steps as in the training phase - This ensures consistency between training and inference.
    input_df['Gender'] = input_df['Gender'].map({'Female': 0, 'Male': 1})
    input_df['OverTime'] = input_df['OverTime'].map({'Yes': 1, 'No': 0})
    ohe_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    input_df_encoded = pd.get_dummies(input_df, columns=ohe_cols, drop_first=True)

    input_df_encoded['TenureCategory'] = input_df_encoded['YearsAtCompany'].apply(tenure_group_input)
    if set(['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']).issubset(input_df_encoded.columns):
        input_df_encoded['EngagementScore'] = input_df_encoded[['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    input_df_encoded['ExperienceLevel'] = input_df_encoded['TotalWorkingYears'].apply(experience_group_input)
    input_df_encoded['PromotionGapFlag'] = input_df_encoded['YearsSinceLastPromotion'].apply(lambda x: 1 if x >= 5 else 0)
    input_df_final = pd.get_dummies(input_df_encoded, columns=['TenureCategory', 'ExperienceLevel'], drop_first=True)

    # Reindex the final input DataFrame to match the order of features the model expects. Missing features will be added with a value of 0.
    input_df_final = input_df_final.reindex(columns=expected_features, fill_value=0)

    return input_df_final

# Call the user input function to get the processed features
input_df = user_input_features()
# -----------------------------------------

# Step 6.7: Model Prediction and Display Results - Performs prediction using the loaded model and displays the attrition risk.
st.markdown("===========================================================================")
with st.container():
    st.subheader("📊 Employee Attrition Risk Analysis:")
    try:
        # Scale the input features using the loaded scaler
        scaled_input = scaler.transform(input_df)
        # Make a prediction using the loaded model
        prediction = model.predict(scaled_input)
        # Get the probability of attrition (class 1)
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ Employee is likely to leave! (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Employee is likely to stay. (Probability: {probability:.2f})")

    except ValueError as e:
        st.error(f"Error during scaling or prediction: {e}")
        st.warning("Please check the input values and ensure the model and scaler files are correctly loaded and compatible.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.warning("Please verify your data file path and the integrity of the saved model/scaler files.")
        st.stop()

    # Display the input details provided by the user in an expandable section.
    with st.expander("📋 Click to view Employee Details Provided:"):
        st.write(input_df)
# -----------------------------------------


# -----------------------------------------
# Stage 7: Additional Dashboards and Graphs (EDA Visualizations)
# -----------------------------------------
st.markdown("===========================================================================")
st.header("📊 Attrition Trends & Key Factors")
st.markdown("Predict whether an employee is likely to leave or stay based on various HR parameters.")

# Load the original dataset again specifically for plotting. This ensures plots use original, unencoded categorical names for better readability.
plot_df = pd.read_csv(file_path)
plot_df['Attrition'] = plot_df['Attrition'].map({'Yes': 1, 'No': 0})

# List to store all generated plot figures and their titles
all_plots = []

# Helper function to add bar labels for countplots and barplots
def add_bar_labels(ax, orient='v', fmt='%d'):
    if orient == 'v':
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, fontsize=6, fontweight='bold', color='gray')
    elif orient == 'h':
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, fontsize=6, fontweight='bold', color='gray')


# Step 7.1: Overall Attrition Distribution
fig_overall_attrition, ax_overall_attrition = plt.subplots(figsize=(7, 5))
sns.countplot(x='Attrition', data=plot_df, ax=ax_overall_attrition, palette='viridis')
ax_overall_attrition.set_title("Overall Attrition Count (0 = Stayed, 1 = Left)", fontsize=10, fontweight='bold', color='blue')
ax_overall_attrition.set_xlabel("Attrition Status", fontsize=8, fontweight='bold', color='red')
ax_overall_attrition.set_ylabel("Number of Employees", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_overall_attrition)
all_plots.append((fig_overall_attrition, "Overall Attrition Distribution"))
# -----------------------------------------

# Step 7.2: Age Distribution by Attrition
fig_age_dist, ax_age_dist = plt.subplots(figsize=(7, 5))
sns.histplot(data=plot_df, x='Age', hue='Attrition', kde=True, ax=ax_age_dist, palette='plasma', stat='density', common_norm=False) # Changed palette
ax_age_dist.set_title("Age Distribution by Attrition", fontsize=10, fontweight='bold', color='blue')
ax_age_dist.set_xlabel("Age", fontsize=8, fontweight='bold', color='red')
ax_age_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_age_dist, "Age Distribution by Attrition"))
# -----------------------------------------

# Step 7.3: Attrition Percentage by Age Group
bins = [18, 25, 35, 45, 55, 65]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64']
plot_df['AgeGroup'] = pd.cut(plot_df['Age'], bins=bins, labels=labels, right=False)
fig_age_group_attrition, ax_age_group_attrition = plt.subplots(figsize=(7, 5))
sns.barplot(x='AgeGroup', y='Attrition', hue='Attrition', data=plot_df, ax=ax_age_group_attrition, palette='flare', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_age_group_attrition.set_title("Attrition Percentage by Age Group", fontsize=10, fontweight='bold', color='blue')
ax_age_group_attrition.set_xlabel("Age Group", fontsize=8, fontweight='bold', color='red')
ax_age_group_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_age_group_attrition, fmt='%.1f%%')
all_plots.append((fig_age_group_attrition, "Attrition Percentage by Age Group"))
# -----------------------------------------

# Step 7.4: Attrition by Business Travel
fig_business_travel_attrition, ax_business_travel_attrition = plt.subplots(figsize=(8, 6))
sns.barplot(x='BusinessTravel', y='Attrition', hue='Attrition', data=plot_df, ax=ax_business_travel_attrition, palette='magma', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_business_travel_attrition.set_title("Attrition Percentage by Business Travel", fontsize=10, fontweight='bold', color='blue')
ax_business_travel_attrition.set_xlabel("Business Travel", fontsize=8, fontweight='bold', color='red')
ax_business_travel_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_business_travel_attrition, fmt='%.1f%%')
all_plots.append((fig_business_travel_attrition, "Attrition Percentage by Business Travel"))
# -----------------------------------------

# Step 7.5: Daily Rate Distribution by Attrition
fig_daily_rate_dist, ax_daily_rate_dist = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=plot_df, x='DailyRate', hue='Attrition', fill=True, common_norm=False, ax=ax_daily_rate_dist, palette='cividis') # Changed palette
ax_daily_rate_dist.set_title("Daily Rate Distribution by Attrition (KDE)", fontsize=10, fontweight='bold', color='blue')
ax_daily_rate_dist.set_xlabel("Daily Rate", fontsize=8, fontweight='bold', color='red')
ax_daily_rate_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_daily_rate_dist, "Daily Rate Distribution by Attrition"))
# -----------------------------------------

# Step 7.6: Attrition by Department
dept_attrition = pd.crosstab(plot_df['Department'], plot_df['Attrition'], normalize='index') * 100
fig_dept_attrition, ax_dept_attrition = plt.subplots(figsize=(7, 5))
dept_attrition.plot(kind='bar', stacked=True, ax=ax_dept_attrition, colormap='rocket') # Changed colormap
ax_dept_attrition.set_title("Attrition Percentage by Department", fontsize=10, fontweight='bold', color='blue')
ax_dept_attrition.set_xlabel("Department", fontsize=8, fontweight='bold', color='red')
ax_dept_attrition.set_ylabel("Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_dept_attrition.tick_params(axis='x', rotation=45)
ax_dept_attrition.legend(title='Attrition', labels=['Stayed', 'Left'])
all_plots.append((fig_dept_attrition, "Attrition Percentage by Department"))
# -----------------------------------------

# Step 7.7: Distance From Home Distribution by Attrition
fig_distance_from_home_dist, ax_distance_from_home_dist = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=plot_df, x='DistanceFromHome', hue='Attrition', fill=True, common_norm=False, ax=ax_distance_from_home_dist, palette='mako') # Changed palette
ax_distance_from_home_dist.set_title("Distance From Home Distribution by Attrition (KDE)", fontsize=10, fontweight='bold', color='blue')
ax_distance_from_home_dist.set_xlabel("Distance From Home", fontsize=8, fontweight='bold', color='red')
ax_distance_from_home_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_distance_from_home_dist, "Distance From Home Distribution by Attrition"))
# -----------------------------------------

# Step 7.8: Attrition by Education Level
fig_education_attrition, ax_education_attrition = plt.subplots(figsize=(7, 5))
sns.barplot(x='Education', y='Attrition', hue='Attrition', data=plot_df, ax=ax_education_attrition, palette='crest', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_education_attrition.set_title("Attrition Percentage by Education Level", fontsize=10, fontweight='bold', color='blue')
ax_education_attrition.set_xlabel("Education Level (1=Below College, 5=Doctor)", fontsize=8, fontweight='bold', color='red')
ax_education_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_education_attrition, fmt='%.1f%%')
all_plots.append((fig_education_attrition, "Attrition Percentage by Education Level"))
# -----------------------------------------

# Step 7.9: Attrition by Education Field
edu_field_attrition = pd.crosstab(plot_df['EducationField'], plot_df['Attrition'], normalize='index') * 100
fig_edu_field_attrition, ax_edu_field_attrition = plt.subplots(figsize=(8, 6))
edu_field_attrition.plot(kind='bar', stacked=True, ax=ax_edu_field_attrition, colormap='cubehelix') # Changed colormap
ax_edu_field_attrition.set_title("Attrition Percentage by Education Field", fontsize=10, fontweight='bold', color='blue')
ax_edu_field_attrition.set_xlabel("Education Field", fontsize=8, fontweight='bold', color='red')
ax_edu_field_attrition.set_ylabel("Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_edu_field_attrition.tick_params(axis='x', rotation=45)
ax_edu_field_attrition.legend(title='Attrition', labels=['Stayed', 'Left'])
all_plots.append((fig_edu_field_attrition, "Attrition Percentage by Education Field"))
# -----------------------------------------

# Step 7.10: Attrition Distribution by Gender (Pie Charts)
gender_attrition_counts = plot_df.groupby('Gender')['Attrition'].value_counts(normalize=True).unstack() * 100
fig_gender_attrition, (ax_gender_attrition_f, ax_gender_attrition_m) = plt.subplots(1, 2, figsize=(10, 5))
if 'Female' in gender_attrition_counts.index:
    ax_gender_attrition_f.pie(gender_attrition_counts.loc['Female'], labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90, colors=['#008080', '#DC143C']) # Darker/Brighter colors
    ax_gender_attrition_f.set_title("Attrition for Female Employees", fontsize=10, fontweight='bold', color='blue')
    ax_gender_attrition_f.axis('equal')
if 'Male' in gender_attrition_counts.index:
    ax_gender_attrition_m.pie(gender_attrition_counts.loc['Male'], labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90, colors=['#4682B4', '#B22222']) # Darker/Brighter colors
    ax_gender_attrition_m.set_title("Attrition for Male Employees", fontsize=10, fontweight='bold', color='blue')
    ax_gender_attrition_m.axis('equal')
fig_gender_attrition.suptitle("Attrition Distribution by Gender", fontsize=12, fontweight='bold', color='blue')
all_plots.append((fig_gender_attrition, "Attrition Distribution by Gender"))
# -----------------------------------------

# Step 7.11: Hourly Rate Distribution by Attrition
fig_hourly_rate_dist, ax_hourly_rate_dist = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=plot_df, x='HourlyRate', hue='Attrition', fill=True, common_norm=False, ax=ax_hourly_rate_dist, palette='viridis') # Changed palette
ax_hourly_rate_dist.set_title("Hourly Rate Distribution by Attrition (KDE)", fontsize=10, fontweight='bold', color='blue')
ax_hourly_rate_dist.set_xlabel("Hourly Rate", fontsize=8, fontweight='bold', color='red')
ax_hourly_rate_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_hourly_rate_dist, "Hourly Rate Distribution by Attrition"))
# -----------------------------------------

# Step 7.12: Attrition by Job Involvement
involvement_attrition = pd.crosstab(plot_df['JobInvolvement'], plot_df['Attrition'], normalize='index') * 100
fig_job_involvement_attrition, ax_job_involvement_attrition = plt.subplots(figsize=(7, 5))
involvement_attrition.plot(kind='bar', stacked=True, ax=ax_job_involvement_attrition, colormap='Spectral') # Changed colormap
ax_job_involvement_attrition.set_title("Attrition Percentage by Job Involvement", fontsize=10, fontweight='bold', color='blue')
ax_job_involvement_attrition.set_xlabel("Job Involvement (1=Low, 4=Very High)", fontsize=8, fontweight='bold', color='red')
ax_job_involvement_attrition.set_ylabel("Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_job_involvement_attrition.tick_params(axis='x', rotation=0)
ax_job_involvement_attrition.legend(title='Attrition', labels=['Stayed', 'Left'])
all_plots.append((fig_job_involvement_attrition, "Attrition Percentage by Job Involvement"))
# -----------------------------------------

# Step 7.13: Attrition Percentage by Job Level
fig_job_level_attrition, ax_job_level_attrition = plt.subplots(figsize=(7, 5))
sns.barplot(x='JobLevel', y='Attrition', hue='Attrition', data=plot_df, ax=ax_job_level_attrition, palette='coolwarm', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_job_level_attrition.set_title("Attrition Percentage by Job Level", fontsize=10, fontweight='bold', color='blue')
ax_job_level_attrition.set_xlabel("Job Level", fontsize=8, fontweight='bold', color='red')
ax_job_level_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_job_level_attrition, fmt='%.1f%%')
all_plots.append((fig_job_level_attrition, "Attrition Percentage by Job Level"))
# -----------------------------------------

# Step 7.14: Attrition by Job Role (Sorted by Attrition Rate)
original_df_for_plot_jobrole = pd.read_csv(file_path)
original_df_for_plot_jobrole['Attrition'] = original_df_for_plot_jobrole['Attrition'].map({'Yes': 1, 'No': 0})
job_role_attrition_percent = original_df_for_plot_jobrole.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack() * 100
job_role_attrition_percent_sorted = job_role_attrition_percent.sort_values(by=1, ascending=False).index.tolist()
fig_job_role_attrition, ax_job_role_attrition = plt.subplots(figsize=(10, 7))
sns.barplot(y='JobRole', x='Attrition', hue='Attrition', data=original_df_for_plot_jobrole, order=job_role_attrition_percent_sorted, ax=ax_job_role_attrition, palette='RdBu', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_job_role_attrition.set_title("Attrition Percentage by Job Role (Sorted by Attrition Rate)", fontsize=10, fontweight='bold', color='blue')
ax_job_role_attrition.set_xlabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_job_role_attrition.set_ylabel("Job Role", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_job_role_attrition, orient='h', fmt='%.1f%%')
all_plots.append((fig_job_role_attrition, "Attrition Percentage by Job Role"))
# -----------------------------------------

# Step 7.15: Job Role vs. Monthly Income by Attrition
fig_job_role_income_attrition, ax_job_role_income_attrition = plt.subplots(figsize=(10, 7))
sns.boxenplot(x='Attrition', y='MonthlyIncome', hue='JobRole', data=plot_df, ax=ax_job_role_income_attrition, palette='Paired') # Changed palette
ax_job_role_income_attrition.set_title("Job Role vs. Monthly Income by Attrition", fontsize=10, fontweight='bold', color='blue')
ax_job_role_income_attrition.set_xlabel("Attrition (0=Stayed, 1=Left)", fontsize=8, fontweight='bold', color='red')
ax_job_role_income_attrition.set_ylabel("Monthly Income", fontsize=8, fontweight='bold', color='red')
ax_job_role_income_attrition.tick_params(axis='x', rotation=0)
ax_job_role_income_attrition.legend(title='Job Role', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
all_plots.append((fig_job_role_income_attrition, "Job Role vs. Monthly Income by Attrition"))
# -----------------------------------------

# Step 7.16: Attrition by Job Satisfaction
fig_job_satisfaction_attrition, ax_job_satisfaction_attrition = plt.subplots(figsize=(7, 5))
sns.barplot(x='JobSatisfaction', y='Attrition', hue='Attrition', data=plot_df, ax=ax_job_satisfaction_attrition, palette='Set1', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_job_satisfaction_attrition.set_title("Attrition Percentage by Job Satisfaction", fontsize=10, fontweight='bold', color='blue')
ax_job_satisfaction_attrition.set_xlabel("Job Satisfaction (1=Low, 4=Very High)", fontsize=8, fontweight='bold', color='red')
ax_job_satisfaction_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_job_satisfaction_attrition, fmt='%.1f%%')
all_plots.append((fig_job_satisfaction_attrition, "Attrition Percentage by Job Satisfaction"))
# -----------------------------------------

# Step 7.17: Attrition by Marital Status
marital_attrition = pd.crosstab(plot_df['MaritalStatus'], plot_df['Attrition'], normalize='index') * 100
fig_marital_status_attrition, ax_marital_status_attrition = plt.subplots(figsize=(7, 5))
marital_attrition.plot(kind='bar', stacked=True, ax=ax_marital_status_attrition, colormap='Set2') # Changed colormap
ax_marital_status_attrition.set_title("Attrition Percentage by Marital Status", fontsize=10, fontweight='bold', color='blue')
ax_marital_status_attrition.set_xlabel("Marital Status", fontsize=8, fontweight='bold', color='red')
ax_marital_status_attrition.set_ylabel("Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_marital_status_attrition.tick_params(axis='x', rotation=45)
ax_marital_status_attrition.legend(title='Attrition', labels=['Stayed', 'Left'])
all_plots.append((fig_marital_status_attrition, "Attrition Percentage by Marital Status"))
# -----------------------------------------

# Step 7.18: Monthly Income Distribution by Attrition
fig_monthly_income_dist, ax_monthly_income_dist = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=plot_df, x='MonthlyIncome', hue='Attrition', fill=True, common_norm=False, ax=ax_monthly_income_dist, palette='Set3')
ax_monthly_income_dist.set_title("Monthly Income Distribution by Attrition (KDE)", fontsize=10, fontweight='bold', color='blue')
ax_monthly_income_dist.set_xlabel("Monthly Income", fontsize=8, fontweight='bold', color='red')
ax_monthly_income_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_monthly_income_dist, "Monthly Income Distribution by Attrition"))
# -----------------------------------------

# Step 7.19: Attrition by Number of Companies Worked
fig_num_companies_worked_attrition, ax_num_companies_worked_attrition = plt.subplots(figsize=(7, 5))
sns.barplot(x='NumCompaniesWorked', y='Attrition', hue='Attrition', data=plot_df, ax=ax_num_companies_worked_attrition, palette='Dark2', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_num_companies_worked_attrition.set_title("Attrition Percentage by Number of Companies Worked", fontsize=10, fontweight='bold', color='blue')
ax_num_companies_worked_attrition.set_xlabel("Number of Companies Worked", fontsize=8, fontweight='bold', color='red')
ax_num_companies_worked_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_num_companies_worked_attrition, fmt='%.1f%%')
all_plots.append((fig_num_companies_worked_attrition, "Attrition Percentage by Number of Companies Worked"))
# -----------------------------------------

# Step 7.20: Attrition by Overtime (Pie Charts)
overtime_yes = plot_df[plot_df['OverTime'] == 'Yes']['Attrition'].value_counts(normalize=True) * 100
overtime_no = plot_df[plot_df['OverTime'] == 'No']['Attrition'].value_counts(normalize=True) * 100

fig_overtime_attrition, (ax_overtime_yes, ax_overtime_no) = plt.subplots(1, 2, figsize=(10, 5))
ax_overtime_yes.pie(overtime_yes, labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90, colors=['#008B8B', '#B22222']) # Darker/Brighter colors
ax_overtime_yes.set_title("Attrition for Overtime: Yes", fontsize=10, fontweight='bold', color='blue')
ax_overtime_yes.axis('equal')

ax_overtime_no.pie(overtime_no, labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90, colors=['#228B22', '#FF4500']) # Darker/Brighter colors
ax_overtime_no.set_title("Attrition for Overtime: No", fontsize=10, fontweight='bold', color='blue')
ax_overtime_no.axis('equal')
fig_overtime_attrition.suptitle("Attrition Distribution by Overtime", fontsize=12, fontweight='bold', color='blue')
all_plots.append((fig_overtime_attrition, "Attrition Distribution by Overtime"))
# -----------------------------------------

# Step 7.21: Percent Salary Hike Distribution by Attrition
fig_salary_hike_dist, ax_salary_hike_dist = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=plot_df, x='PercentSalaryHike', hue='Attrition', fill=True, common_norm=False, ax=ax_salary_hike_dist, palette='Accent')
ax_salary_hike_dist.set_title("Percent Salary Hike Distribution by Attrition (KDE)", fontsize=10, fontweight='bold', color='blue')
ax_salary_hike_dist.set_xlabel("Percent Salary Hike (%)", fontsize=8, fontweight='bold', color='red')
ax_salary_hike_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_salary_hike_dist, "Percent Salary Hike Distribution by Attrition"))
# -----------------------------------------

# Step 7.22: Attrition by Performance Rating
fig_performance_rating_attrition, ax_performance_rating_attrition = plt.subplots(figsize=(7, 5))
sns.barplot(x='PerformanceRating', y='Attrition', hue='Attrition', data=plot_df, ax=ax_performance_rating_attrition, palette='tab10', estimator=lambda x: sum(x)/len(x)*100, errorbar=None) # Changed palette
ax_performance_rating_attrition.set_title("Attrition by Performance Rating", fontsize=10, fontweight='bold', color='blue')
ax_performance_rating_attrition.set_xlabel("Performance Rating", fontsize=8, fontweight='bold', color='red')
ax_performance_rating_attrition.set_ylabel("Attrition Percentage (%)", fontsize=8, fontweight='bold', color='red')
add_bar_labels(ax_performance_rating_attrition, fmt='%.1f%%')
all_plots.append((fig_performance_rating_attrition, "Attrition Percentage by Performance Rating"))
# -----------------------------------------

# Step 7.23: Attrition by Stock Option Level
stock_attrition = pd.crosstab(plot_df['StockOptionLevel'], plot_df['Attrition'], normalize='index') * 100
fig_stock_option_attrition, ax_stock_option_attrition = plt.subplots(figsize=(7, 5))
stock_attrition.plot(kind='bar', stacked=True, ax=ax_stock_option_attrition, colormap='tab20') # Changed colormap
ax_stock_option_attrition.set_title("Attrition Percentage by Stock Option Level", fontsize=10, fontweight='bold', color='blue')
ax_stock_option_attrition.set_xlabel("Stock Option Level", fontsize=8, fontweight='bold', color='red')
ax_stock_option_attrition.set_ylabel("Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_stock_option_attrition.tick_params(axis='x', rotation=0)
ax_stock_option_attrition.legend(title='Attrition', labels=['Stayed', 'Left'])
all_plots.append((fig_stock_option_attrition, "Attrition Percentage by Stock Option Level"))
# -----------------------------------------

# Step 7.24: Total Working Years Distribution by Attrition
fig_total_working_years_dist, ax_total_working_years_dist = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=plot_df, x='TotalWorkingYears', hue='Attrition', fill=True, common_norm=False, ax=ax_total_working_years_dist, palette='viridis') # Changed palette
ax_total_working_years_dist.set_title("Total Working Years Distribution by Attrition (KDE)", fontsize=10, fontweight='bold', color='blue')
ax_total_working_years_dist.set_xlabel("Total Working Years", fontsize=8, fontweight='bold', color='red')
ax_total_working_years_dist.set_ylabel("Density", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_total_working_years_dist, "Total Working Years Distribution by Attrition"))
# -----------------------------------------

# Step 7.25: Attrition by Training Times Last Year
training_attrition = pd.crosstab(plot_df['TrainingTimesLastYear'], plot_df['Attrition'], normalize='index') * 100
fig_training_times_attrition, ax_training_times_attrition = plt.subplots(figsize=(7, 5))
training_attrition.plot(kind='bar', stacked=True, ax=ax_training_times_attrition, colormap='rocket') # Changed colormap
ax_training_times_attrition.set_title("Attrition Percentage by Training Times Last Year", fontsize=10, fontweight='bold', color='blue')
ax_training_times_attrition.set_xlabel("Training Times Last Year", fontsize=8, fontweight='bold', color='red')
ax_training_times_attrition.set_ylabel("Percentage (%)", fontsize=8, fontweight='bold', color='red')
ax_training_times_attrition.tick_params(axis='x', rotation=0)
ax_training_times_attrition.legend(title='Attrition', labels=['Stayed', 'Left'])
all_plots.append((fig_training_times_attrition, "Attrition Distribution by Training Times Last Year"))
# -----------------------------------------

# Step 7.26: Work-Life Balance Distribution by Attrition (Violin Plot with Age)
fig_work_life_balance_dist, ax_work_life_balance_dist = plt.subplots(figsize=(8, 6))
sns.violinplot(x='WorkLifeBalance', y='Age', hue='Attrition', data=plot_df, ax=ax_work_life_balance_dist, palette='plasma', split=True) # Changed palette
ax_work_life_balance_dist.set_title("Work-Life Balance Distribution by Attrition (Violin Plot)", fontsize=10, fontweight='bold', color='blue')
ax_work_life_balance_dist.set_xlabel("Work-Life Balance (1=Bad, 4=Best)", fontsize=8, fontweight='bold', color='red')
ax_work_life_balance_dist.set_ylabel("Age", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_work_life_balance_dist, "Work-Life Balance Distribution by Attrition"))
# -----------------------------------------

# Step 7.27: Years at Company vs. Attrition
fig_years_at_company_attrition, ax_years_at_company_attrition = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Attrition', y='YearsAtCompany', data=plot_df, ax=ax_years_at_company_attrition, palette='viridis') # Changed palette
ax_years_at_company_attrition.set_title("Years at Company vs. Attrition", fontweight='bold', color='blue')
ax_years_at_company_attrition.set_xlabel("Attrition (0=Stayed, 1=Left)", fontsize=8, fontweight='bold', color='red')
ax_years_at_company_attrition.set_ylabel("Years at Company", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_years_at_company_attrition, "Years at Company vs. Attrition"))
# -----------------------------------------

# Step 7.28: Years In Current Role Distribution by Attrition
fig_years_in_current_role_dist, ax_years_in_current_role_dist = plt.subplots(figsize=(8, 6))
sns.violinplot(x='Attrition', y='YearsInCurrentRole', data=plot_df, ax=ax_years_in_current_role_dist, palette='mako', hue='Attrition', legend=False, inner='quartile') # Changed palette
ax_years_in_current_role_dist.set_title("Years In Current Role Distribution by Attrition", fontsize=10, fontweight='bold', color='blue')
ax_years_in_current_role_dist.set_xlabel("Attrition (0=Stayed, 1=Left)", fontsize=8, fontweight='bold', color='red')
ax_years_in_current_role_dist.set_ylabel("Years In Current Role", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_years_in_current_role_dist, "Years In Current Role Distribution by Attrition"))
# -----------------------------------------

# Step 7.29: Years Since Last Promotion Distribution by Attrition
fig_years_since_last_promotion_dist, ax_years_since_last_promotion_dist = plt.subplots(figsize=(8, 6))
sns.violinplot(x='Attrition', y='YearsSinceLastPromotion', data=plot_df, ax=ax_years_since_last_promotion_dist, palette='flare', hue='Attrition', legend=False, inner='quartile')
ax_years_since_last_promotion_dist.set_title("Years Since Last Promotion Distribution by Attrition", fontsize=10, fontweight='bold', color='blue')
ax_years_since_last_promotion_dist.set_xlabel("Attrition (0=Stayed, 1=Left)", fontsize=8, fontweight='bold', color='red')
ax_years_since_last_promotion_dist.set_ylabel("Years Since Last Promotion", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_years_since_last_promotion_dist, "Years Since Last Promotion Distribution by Attrition"))
# -----------------------------------------

# Step 7.30: Years With Current Manager Distribution by Attrition
fig_years_with_curr_manager_dist, ax_years_with_curr_manager_dist = plt.subplots(figsize=(8, 6))
sns.violinplot(x='Attrition', y='YearsWithCurrManager', data=plot_df, ax=ax_years_with_curr_manager_dist, palette='crest', hue='Attrition', legend=False, inner='quartile') # Changed palette
ax_years_with_curr_manager_dist.set_title("Years With Current Manager Distribution by Attrition", fontsize=10, fontweight='bold', color='blue')
ax_years_with_curr_manager_dist.set_xlabel("Attrition (0=Stayed, 1=Left)", fontsize=8, fontweight='bold', color='red')
ax_years_with_curr_manager_dist.set_ylabel("Years With Current Manager", fontsize=8, fontweight='bold', color='red')
all_plots.append((fig_years_with_curr_manager_dist, "Years With Current Manager Distribution by Attrition"))
# -----------------------------------------

# Step 7.31: Correlation Heatmap of Numerical Features
fig_correlation_heatmap, ax_correlation_heatmap = plt.subplots(figsize=(12, 10))
numerical_df = plot_df.select_dtypes(include=np.number)
sns.heatmap(numerical_df.corr(), annot=True, cmap='RdBu', fmt=".2f", linewidths=.5, ax=ax_correlation_heatmap)
ax_correlation_heatmap.set_title("Correlation Heatmap of Numerical Features", fontsize=10, fontweight='bold', color='blue')
ax_correlation_heatmap.tick_params(axis='x', rotation=90)
ax_correlation_heatmap.tick_params(axis='y', rotation=0)
plt.tight_layout()
all_plots.append((fig_correlation_heatmap, "Correlation Heatmap of Numerical Features"))
# -----------------------------------------

# Step 7.32: Arranges the generated plots in a multi-column layout for better dashboard presentation.

col1, col2, col3, col4, col5 = st.columns(5)

for i, (plot_fig, plot_title) in enumerate(all_plots):
    if i % 5 == 0: # First column
        with col1:
            st.pyplot(plot_fig)
    elif i % 5 == 1: # Second column
        with col2:
            st.pyplot(plot_fig)
    elif i % 5 == 2: # Third column
        with col3:
            st.pyplot(plot_fig)
    elif i % 5 == 3: # Fourth column
        with col4:
            st.pyplot(plot_fig)
    else: # Fifth column
        with col5:
            st.pyplot(plot_fig)
    plt.close(plot_fig)
# -----------------------------------------

# -----------------------------------------
# Stage 8: Future Enhancements
# -----------------------------------------

st.markdown("===========================================================================")
st.header("🏃🏻‍♀️‍➡️ ⚽︎ 🤝Future Enhancements🤝 ⚽︎ 🏃🏻‍♀️")

# Define the enhancements and their descriptions
enhancements = {
    "Predict Performance Rating": {
        "description": """
This feature would involve training a separate **regression model** (e.g., Linear Regression, Random Forest Regressor, Gradient Boosting Regressor) to predict an employee's Performance Rating based on other relevant features.
""",
        "info": "To implement: Train a regression model, save it, and create a new input/prediction section in the Streamlit app similar to the Attrition Prediction."
    },
    "Predict Promotion Likelihood": {
        "description": """
The `PromotionGapFlag` feature has been engineered, indicating if an employee has not been promoted in 5+ years. A dedicated **classification model** could be trained to predict the likelihood of an employee receiving a promotion in the near future, using features like `JobLevel`, `TotalWorkingYears`, `YearsSinceLastPromotion`, and `PerformanceRating`.
""",
        "info": "To implement: Train a new classification model (e.g., Logistic Regression or another Random Forest) specifically for promotion prediction. Integrate its prediction into the app."
    },
    "Multi-model comparison with hyperparameter tuning": {
        "description": """
While a basic `GridSearchCV` example for Random Forest is included in the training script, a full multi-model comparison would involve:
* Evaluating several different machine learning algorithms (e.g., SVM, Gradient Boosting, XGBoost).
* Performing comprehensive hyperparameter tuning for each model using techniques like `GridSearchCV` or `RandomizedSearchCV`.
* Comparing their performance metrics (accuracy, precision, recall, F1-score, ROC-AUC) to select the best-performing model.
""",
        "info": "The training script (Part 1) now includes a basic `GridSearchCV` example for Random Forest. Further enhancements would involve extending this to more models and potentially more complex tuning strategies."
    },
    "Job Satisfaction Prediction (Regression Model)": {
        "description": """
This would involve predicting an employee's Job Satisfaction score (a numerical value) based on other features like `EnvironmentSatisfaction`, `RelationshipSatisfaction`, `MonthlyIncome`, `JobRole`, etc. This requires a **regression model**.
""",
        "info": "To implement: Train a regression model (e.g., `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`) with 'JobSatisfaction' as the target variable. Save the model and integrate its prediction into a new section of the Streamlit app."
    },
    "Live Model Explanations (SHAP/LIME)": {
        "description": """
Implementing live model explanations using libraries like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) would provide insights into *why* the model made a particular prediction for a given employee. This enhances model interpretability and trustworthiness.
""",
        "info": "To implement: Install `shap` or `lime` libraries. Integrate their explanation functions into the Streamlit app to generate and visualize feature importance for individual predictions."
    }
}

# Create a dropdown (select box) for the enhancements, centered using columns
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    selected_enhancement = st.selectbox(
        "Select a future enhancement to learn more:",
        list(enhancements.keys()),
        index=0 # Default to the first option
    )

# Display the description and info for the selected enhancement
if selected_enhancement:
    st.subheader(selected_enhancement)
    st.write(enhancements[selected_enhancement]["description"])
    st.info(enhancements[selected_enhancement]["info"])
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Predictive Analysis of Oil and Gas Wells",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
.main-header {
    font-size: 3em;
    font-weight: bold;
    color: #2E86C1;
    text-align: center;
    margin-bottom: 30px;
}
.section-header {
    font-size: 2em;
    font-weight: bold;
    color: #2874A6;
    margin-top: 20px;
    margin-bottom: 15px;
}
.stButton>button {
    background-color: #3498DB;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 5px;
}
.stFileUploader label {
    font-weight: bold;
    color: #2874A6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class=\'main-header\'>Predictive Analysis of Oil and Gas Wells Using Decision Tree</h1>", unsafe_allow_html=True)
st.markdown("<h3 style=\'text-align: center; color: #5D6D7E;\'>A Case Study of New York Production Data</h3>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Upload Dataset",
    "Data Overview",
    "Data Exploration",
    "Preprocessing",
    "Modeling",
    "New Data Prediction"
])

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'best_dt' not in st.session_state:
    st.session_state.best_dt = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# --- Page: Upload Dataset ---
if page == "Upload Dataset":
    st.markdown("<h2 class=\'section-header\'>Upload Dataset</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("Dataset uploaded successfully!")
            st.write("First 5 rows of the dataset:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- Page: Data Overview ---
elif page == "Data Overview":
    st.markdown("<h2 class=\'section-header\'>Data Overview</h2>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write("### Dataset Dimensions")
        st.write(f"Number of rows: {df.shape[0]:,}")
        st.write(f"Number of columns: {df.shape[1]}")

        st.write("### Missing Value Statistics")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            st.write("Columns with missing values:")
            st.dataframe(missing_data)

            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data.plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Missing Values per Column', fontweight='bold')
            ax.set_xlabel('Number of Missing Values')
            st.pyplot(fig)
        else:
            st.success("No missing values found in the dataset!")

        st.write("### Data Types Distribution")
        fig, ax = plt.subplots(figsize=(8, 8))
        df.dtypes.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap='Set3')
        ax.set_title('Distribution of Data Types', fontweight='bold')
        ax.set_ylabel('') # Hide the default ylabel
        st.pyplot(fig)

    else:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")

# --- Page: Data Exploration ---
elif page == "Data Exploration":
    st.markdown("<h2 class=\'section-header\'>Data Exploration</h2>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        df = st.session_state.df

        st.write("### Distribution of Well Types")
        # Assuming 'Well Type' is a column in your dataset. Adjust if the column name is different.
        # The original code used 'Production_Flag' as target, so I'll use that as a fallback.
        well_type_col = 'Well Type' if 'Well Type' in df.columns else 'Production_Flag'

        if well_type_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, y=well_type_col, palette='viridis', ax=ax)
            ax.set_title(f'Distribution of {well_type_col}', fontweight='bold')
            ax.set_xlabel('Count')
            ax.set_ylabel(well_type_col)
            st.pyplot(fig)
        else:
            st.warning(f"Column '{well_type_col}' not found. Please ensure your dataset has a '{well_type_col}' column for this visualization or adjust the column name.")

        st.write("### Geographical Distribution of Wells")
        # Assuming 'Latitude' and 'Longitude' columns exist
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            st.map(df[['Latitude', 'Longitude']].dropna())
        else:
            st.warning("Columns 'Latitude' and/or 'Longitude' not found. Cannot display geographical distribution.")

        st.write("### Statistics of Important Columns")
        # Adjust column names as per your dataset. Using common names as examples.
        important_cols = ['Operator', 'Formation', 'Well Status', 'County', 'API_WellNo'] 
        for col in important_cols:
            if col in df.columns:
                st.write(f"#### {col} Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                # Handle cases where value_counts might be too large or have non-string types
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    df[col].value_counts().nlargest(10).plot(kind='barh', ax=ax, color='teal')
                else:
                    # For numerical columns, show a histogram or describe basic stats
                    st.write(df[col].describe())
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax, color='teal')
                
                ax.set_title(f'Top 10 {col} Distribution' if (df[col].dtype == 'object' or df[col].dtype == 'category') else f'Distribution of {col}', fontweight='bold')
                ax.set_xlabel('Count' if (df[col].dtype == 'object' or df[col].dtype == 'category') else 'Value')
                ax.set_ylabel(col)
                st.pyplot(fig)
            else:
                st.warning(f"Column '{col}' not found in the dataset.")

    else:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")

# --- Page: Preprocessing ---
elif page == "Preprocessing":
    st.markdown("<h2 class=\'section-header\'>Preprocessing</h2>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        st.write("### Select Target Column")
        all_columns = df.columns.tolist()
        # Try to pre-select 'Production_Flag' or 'Well Type' if they exist
        default_target_index = 0
        if 'Production_Flag' in all_columns:
            default_target_index = all_columns.index('Production_Flag')
        elif 'Well Type' in all_columns:
            default_target_index = all_columns.index('Well Type')

        target_col = st.selectbox("Choose the target column (e.g., Well Type)", all_columns, index=default_target_index)
        st.session_state.target_col = target_col

        if st.button("Perform Preprocessing"):
            with st.spinner("Preprocessing data..."):
                # Before preprocessing stats
                before_shape = df.shape
                before_missing = df.isnull().sum().sum()

                # Drop missing target
                df_clean = df.dropna(subset=[target_col]).copy()

                # Handle numeric columns
                numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
                if target_col in numeric_cols:
                    numeric_cols = numeric_cols.drop(target_col)

                # Fill missing numeric values with median
                for col in numeric_cols:
                    if df_clean[col].isnull().sum() > 0:
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)

                # Handle categorical columns
                categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
                label_encoders = {}

                for col in categorical_cols:
                    if col != target_col:
                        le = LabelEncoder()
                        # Ensure all values are strings before fitting, to avoid errors with mixed types
                        df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
                        le.fit(df_clean[col])
                        df_clean[col] = le.transform(df_clean[col])
                        label_encoders[col] = le

                # After preprocessing stats
                after_shape = df_clean.shape
                after_missing = df_clean.isnull().sum().sum()

                st.session_state.df_clean = df_clean
                st.session_state.label_encoders = label_encoders

                st.success("Preprocessing complete!")

                st.write("### Preprocessing Summary")
                st.write(f"Original Data Shape: {before_shape}")
                st.write(f"Cleaned Data Shape: {after_shape}")
                st.write(f"Missing Values (Before): {before_missing:,}")
                st.write(f"Missing Values (After): {after_missing:,}")

                # Visualization of preprocessing results
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                categories = ['Number of Rows', 'Number of Columns', 'Missing Values']
                before_values = [before_shape[0], before_shape[1], before_missing]
                after_values = [after_shape[0], after_shape[1], after_missing]

                x = np.arange(len(categories))
                width = 0.35

                bars1 = ax.bar(x - width/2, before_values, width, label='Before', color='lightcoral', alpha=0.8)
                bars2 = ax.bar(x + width/2, after_values, width, label='After', color='lightblue', alpha=0.8)

                ax.set_xlabel('Metrics')
                ax.set_ylabel('Count')
                ax.set_title('Comparison Before vs After Preprocessing', fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend()

                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{int(height):,}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom')

                st.pyplot(fig)

    else:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")

# --- Page: Modeling ---
elif page == "Modeling":
    st.markdown("<h2 class=\'section-header\'>Modeling</h2>", unsafe_allow_html=True)
    if st.session_state.df_clean is not None and st.session_state.target_col is not None:
        df_clean = st.session_state.df_clean
        target_col = st.session_state.target_col

        st.write("### Configure Model Training")
        test_size = st.slider("Train/Test Split Ratio (Test Size)", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Random State", value=42, step=1)

        if st.button("Train Decision Tree Model"):
            with st.spinner("Training model..."):
                X = df_clean.drop(columns=[target_col])
                y = df_clean[target_col]

                # Identify classes with only one sample and remove them
                class_counts = y.value_counts()
                single_instance_classes = class_counts[class_counts == 1].index
                if len(single_instance_classes) > 0:
                    st.warning(f"Removing {len(single_instance_classes)} classes with only one sample for stratification: {single_instance_classes.tolist()}")
                    df_filtered = df_clean[~df_clean[target_col].isin(single_instance_classes)].copy()
                    X = df_filtered.drop(columns=[target_col])
                    y = df_filtered[target_col]

                # Check if there are enough samples for stratification after filtering
                if len(y.unique()) > 1 and all(y.value_counts() > 1):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                else:
                    st.error("Not enough samples per class for stratified split after removing single-instance classes. Please check your dataset or consider a different split strategy.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state # Fallback to non-stratified
                    )

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = X.columns.tolist()

                st.write(f"Data split successfully: Training set {X_train.shape[0]} samples, Testing set {X_test.shape[0]} samples.")

                # Hyperparameter tuning
                param_grid = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }

                dt = DecisionTreeClassifier(random_state=random_state)
                grid = GridSearchCV(
                    dt, param_grid, cv=5, scoring='f1_weighted',
                    n_jobs=-1, verbose=0
                )
                grid.fit(X_train, y_train)
                best_dt = grid.best_estimator_
                st.session_state.best_dt = best_dt

                st.success("Model training complete!")

                st.write("### Model Evaluation")
                y_pred = best_dt.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")

                st.write("#### Confusion Matrix")
                # Get unique classes from y_test and y_pred for confusion matrix labels
                unique_classes = np.unique(np.concatenate((y_test, y_pred)))
                cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

                fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    cm, annot=True, fmt='d',
                    xticklabels=[str(c) for c in unique_classes],
                    yticklabels=[str(c) for c in unique_classes],
                    cmap='Blues', cbar_kws={'label': 'Number of Predictions'},
                    square=True, linewidths=0.5, ax=ax_cm
                )
                ax_cm.set_title('Confusion Matrix', fontweight='bold', fontsize=16, pad=20)
                ax_cm.set_xlabel('Predicted Class', fontweight='bold', fontsize=12)
                ax_cm.set_ylabel('Actual Class', fontweight='bold', fontsize=12)
                st.pyplot(fig_cm)

                st.write("#### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.write("### Decision Tree Structure")
                # Visualize the decision tree
                fig_tree, ax_tree = plt.subplots(figsize=(30, 30))
                plot_tree(
                    best_dt,
                    max_depth=3,  # Limit depth for better visualization in Streamlit
                    filled=True,
                    rounded=True,
                    feature_names=st.session_state.feature_names,
                    fontsize=10,
                    class_names=[str(c) for c in best_dt.classes_],
                    proportion=False,
                    impurity=True,
                    precision=2,
                    ax=ax_tree
                )
                ax_tree.set_title('Decision Tree Structure (Max Depth = 3)', fontweight='bold', fontsize=20, pad=20)
                st.pyplot(fig_tree)

                st.write("### Feature Importance")
                importances = pd.Series(best_dt.feature_importances_, index=st.session_state.feature_names)
                top_features = importances.nlargest(15)

                fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax_fi)
                ax_fi.set_title('Top 15 Feature Importance', fontweight='bold')
                ax_fi.set_xlabel('Importance Score')
                ax_fi.set_ylabel('Feature')
                st.pyplot(fig_fi)

    else:
        st.warning("Please upload and preprocess a dataset first in the 'Upload Dataset' and 'Preprocessing' sections.")

# --- Page: New Data Prediction ---
elif page == "New Data Prediction":
    st.markdown("<h2 class=\'section-header\'>New Data Prediction</h2>", unsafe_allow_html=True)
    if st.session_state.best_dt is not None and st.session_state.feature_names is not None and st.session_state.label_encoders is not None:
        st.write("### Input New Well Data")

        input_data = {}
        # Create input fields for each feature used in training
        for feature in st.session_state.feature_names:
            # Try to infer input type based on original DataFrame's dtypes
            original_dtype = None
            if st.session_state.df is not None and feature in st.session_state.df.columns:
                original_dtype = st.session_state.df[feature].dtype

            if feature in st.session_state.label_encoders: # It's a categorical feature that was encoded
                le = st.session_state.label_encoders[feature]
                unique_values = list(le.classes_)
                # Add 'Unknown' if it's not already in the classes (for new unseen categories)
                if 'Unknown' not in unique_values:
                    unique_values.append('Unknown')
                input_data[feature] = st.selectbox(f"Enter {feature}", unique_values)
            elif original_dtype == 'int64' or original_dtype == 'float64':
                # For numeric features, use number_input
                input_data[feature] = st.number_input(f"Enter {feature}", value=0.0, format="%.2f")
            else:
                # Default to text input for other types or if original_dtype is not found
                input_data[feature] = st.text_input(f"Enter {feature}", value="")

        if st.button("Predict Well Type"):
            try:
                # Create a DataFrame from input data
                input_df = pd.DataFrame([input_data])

                # Apply label encoding to categorical features in input_df
                for col, le in st.session_state.label_encoders.items():
                    if col in input_df.columns:
                        # Handle unseen labels during prediction
                        # If a label is unseen, transform it to 'Unknown' and then encode
                        input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                        input_df[col] = le.transform(input_df[col])

                # Ensure all feature columns are present in the input_df, fill missing with 0 or median
                # This part needs careful handling to ensure consistency with training data
                for feature in st.session_state.feature_names:
                    if feature not in input_df.columns:
                        # If a feature was in training but not in input_data (e.g., if user skipped an input)
                        # Fill with a default value (e.g., 0 for numeric, or encoded 'Unknown' for categorical)
                        if st.session_state.df is not None and feature in st.session_state.df.columns:
                            original_dtype = st.session_state.df[feature].dtype
                            if original_dtype in ['int64', 'float64']:
                                input_df[feature] = 0 # Or median from training data if available
                            else: # Categorical
                                if feature in st.session_state.label_encoders:
                                    le = st.session_state.label_encoders[feature]
                                    if 'Unknown' in le.classes_:
                                        input_df[feature] = le.transform(['Unknown'])[0]
                                    else:
                                        input_df[feature] = 0 # Fallback if 'Unknown' not in classes
                                else:
                                    input_df[feature] = 0 # Fallback for unencoded categorical
                        else:
                            input_df[feature] = 0 # General fallback

                # Reorder columns to match training data
                input_df = input_df[st.session_state.feature_names]

                prediction = st.session_state.best_dt.predict(input_df)
                prediction_proba = st.session_state.best_dt.predict_proba(input_df)

                st.write("### Prediction Result")
                # Decode the prediction if the target column was label encoded
                if st.session_state.target_col in st.session_state.label_encoders:
                    target_le = st.session_state.label_encoders[st.session_state.target_col]
                    decoded_prediction = target_le.inverse_transform(prediction)
                    st.success(f"The predicted Well Type is: **{decoded_prediction[0]}**")
                else:
                    st.success(f"The predicted Well Type is: **{prediction[0]}**")

                st.write("Prediction Probabilities:")
                # Ensure class names are correctly displayed for probabilities
                class_names_for_proba = [str(c) for c in st.session_state.best_dt.classes_]
                if st.session_state.target_col in st.session_state.label_encoders:
                    target_le = st.session_state.label_encoders[st.session_state.target_col]
                    class_names_for_proba = [str(c) for c in target_le.inverse_transform(st.session_state.best_dt.classes_)]

                proba_df = pd.DataFrame(prediction_proba, columns=class_names_for_proba)
                st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Please ensure all input fields are correctly filled and the model has been trained.")

    else:
        st.warning("Please train a model first in the 'Modeling' section.")

# Optional: Contextual notes about the oil and gas industry in New York
st.sidebar.markdown("""
---
### About New York Oil & Gas
New York has a long history of oil and gas production, primarily in the western part of the state. The industry has seen shifts over time due to economic factors, technological advancements, and environmental regulations. Understanding well types is crucial for resource management and environmental impact assessment.
""")



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Analisis Prediktif Sumur Minyak dan Gas",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv("oil-and-gas-annual-production-beginning-2001-1.csv")
    return df

@st.cache_data
def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Fill missing values for numerical columns with median
    for col in ["Months in Production", "Gas Produced, Mcf", "Water Produced, bbl", "Oil Produced, bbl"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Fill missing values for categorical columns with mode
    for col in ["County", "Company Name", "API Hole Number", "Sidetrack Code", "Completion Code", 
                "Production Field", "Well Status Code", "Well Name", "Town", "Producing Formation", 
                "New Georeferenced Column"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop rows where 'Well Type Code' is missing
    df.dropna(subset=["Well Type Code"], inplace=True)

    # Feature Engineering: Create a target variable 'Has_Production'
    # Based on 'Well Status Code' - AC means Active
    df["Has_Production"] = (df["Well Status Code"] == "AC").astype(int)

    return df

@st.cache_data
def train_model(df):
    """Train the Decision Tree model"""
    # Select features for the model
    features = ["County", "Well Type Code", "Months in Production", "Gas Produced, Mcf", 
                "Water Produced, bbl", "Oil Produced, bbl", "Reporting Year"]
    X = df[features].copy()
    y = df["Has_Production"]

    # Store original categorical values for later use
    county_encoder = LabelEncoder()
    well_type_encoder = LabelEncoder()
    
    X["County"] = county_encoder.fit_transform(X["County"])
    X["Well Type Code"] = well_type_encoder.fit_transform(X["Well Type Code"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Performance metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test, output_dict=True)

    return best_model, X_train, X_test, y_train, y_test, train_accuracy, test_accuracy, conf_matrix, class_report, county_encoder, well_type_encoder

def main():
    # Title
    st.markdown('<h1 class="main-header">🛢️ Analisis Prediktif Sumur Minyak dan Gas Menggunakan Decision Tree</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Studi Kasus Data Produksi New York</h3>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Sidebar
    st.sidebar.title("📍 Navigasi")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["📘 Deskripsi Aplikasi", "📍 Pilih Wilayah", "📊 Eksplorasi Data", 
         "🔄 Preprocessing", "🧠 Model Training", "🌳 Visualisasi Model", 
         "📈 Evaluasi Model", "📜 Rules Decision Tree"]
    )
    
    if page == "📘 Deskripsi Aplikasi":
        st.markdown('<h2 class="section-header">📘 Deskripsi Aplikasi</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Tujuan Aplikasi
        Aplikasi ini bertujuan untuk **memprediksi apakah suatu wilayah berpotensi memproduksi minyak/gas** 
        menggunakan model **Decision Tree** berdasarkan data historis produksi sumur minyak dan gas di New York.
        
        ### Metode Machine Learning
        - **Algoritma**: Decision Tree Classifier
        - **Hyperparameter Tuning**: GridSearchCV
        - **Fitur**: County, Well Type Code, Months in Production, Gas/Oil/Water Production, Reporting Year
        - **Target**: Status produksi aktif (berdasarkan Well Status Code = "AC")
        
        ### Sumber Data
        Dataset yang digunakan adalah **"Oil and Gas Annual Production Beginning 2001"** dari New York State 
        Department of Environmental Conservation yang berisi informasi tentang:
        - Lokasi sumur (County, Town)
        - Jenis sumur dan status
        - Data produksi minyak, gas, dan air
        - Periode produksi dan tahun pelaporan
        
        ### Fitur Aplikasi
        1. **Eksplorasi Data**: Visualisasi distribusi data dan missing values
        2. **Preprocessing**: Otomatisasi pembersihan data dan feature engineering
        3. **Model Training**: Pelatihan Decision Tree dengan hyperparameter tuning
        4. **Visualisasi Model**: Struktur pohon keputusan dan feature importance
        5. **Evaluasi**: Metrik performa dan analisis overfitting/underfitting
        6. **Interpretabilitas**: Aturan decision tree dalam bentuk teks
        
        ---
        **Catatan**: Model ini hanya untuk tujuan edukasi dan eksplorasi data.
        """)
        
    elif page == "📍 Pilih Wilayah":
        st.markdown('<h2 class="section-header">📍 Pilih Wilayah</h2>', unsafe_allow_html=True)
        
        # Get unique counties
        counties = sorted(df['County'].dropna().unique())
        selected_county = st.selectbox("Pilih County:", counties)
        
        if selected_county:
            county_data = df[df['County'] == selected_county]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_wells = len(county_data)
                st.metric("🏭 Total Sumur", total_wells)
            
            with col2:
                total_gas = county_data['Gas Produced, Mcf'].sum()
                st.metric("⛽ Total Gas (Mcf)", f"{total_gas:,.0f}")
            
            with col3:
                total_oil = county_data['Oil Produced, bbl'].sum()
                st.metric("🛢️ Total Oil (bbl)", f"{total_oil:,.0f}")
            
            with col4:
                years_active = county_data['Reporting Year'].nunique()
                st.metric("📅 Tahun Aktif", years_active)
            
            # Additional statistics
            st.markdown("### 📊 Statistik Detail")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rata-rata Produksi Tahunan:**")
                avg_gas = county_data.groupby('Reporting Year')['Gas Produced, Mcf'].sum().mean()
                avg_oil = county_data.groupby('Reporting Year')['Oil Produced, bbl'].sum().mean()
                st.write(f"- Gas: {avg_gas:,.0f} Mcf/tahun")
                st.write(f"- Oil: {avg_oil:,.0f} bbl/tahun")
                
            with col2:
                st.markdown("**Status Sumur:**")
                status_counts = county_data['Well Status Code'].value_counts()
                for status, count in status_counts.head(5).items():
                    st.write(f"- {status}: {count} sumur")
            
            # Production trend chart
            if len(county_data) > 0:
                st.markdown("### 📈 Trend Produksi")
                yearly_production = county_data.groupby('Reporting Year').agg({
                    'Gas Produced, Mcf': 'sum',
                    'Oil Produced, bbl': 'sum'
                }).reset_index()
                
                fig = px.line(yearly_production, x='Reporting Year', 
                             y=['Gas Produced, Mcf', 'Oil Produced, bbl'],
                             title=f'Trend Produksi di {selected_county}')
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📊 Eksplorasi Data":
        st.markdown('<h2 class="section-header">📊 Eksplorasi Data & Visualisasi</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        st.markdown("### 📋 Overview Dataset")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(df))
        with col2:
            st.metric("📝 Total Columns", len(df.columns))
        with col3:
            st.metric("🗺️ Counties", df['County'].nunique())
        with col4:
            st.metric("📅 Years", df['Reporting Year'].nunique())
        
        # Missing values visualization
        st.markdown("### 🔍 Missing Values Analysis")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.values, y=missing_data.index, 
                        orientation='h', title='Missing Values per Column')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Tidak ada missing values!")
        
        # Data distribution
        st.markdown("### 📊 Distribusi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # County distribution
            county_counts = df['County'].value_counts().head(10)
            fig = px.bar(x=county_counts.values, y=county_counts.index,
                        orientation='h', title='Top 10 Counties by Well Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Well status distribution
            status_counts = df['Well Status Code'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Well Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Production statistics
        st.markdown("### 📈 Statistik Produksi")
        
        production_stats = df[['Gas Produced, Mcf', 'Oil Produced, bbl', 'Water Produced, bbl']].describe()
        st.dataframe(production_stats)
    
    elif page == "🔄 Preprocessing":
        st.markdown('<h2 class="section-header">🔄 Preprocessing Otomatis</h2>', unsafe_allow_html=True)
        
        st.markdown("### 📋 Langkah Preprocessing")
        st.markdown("""
        1. **Handling Missing Values**:
           - Numerical columns: Diisi dengan median
           - Categorical columns: Diisi dengan modus
        
        2. **Feature Engineering**:
           - Target variable: `Has_Production` (berdasarkan Well Status Code = "AC")
           - Features: County, Well Type Code, Months in Production, Gas/Oil/Water Production, Reporting Year
        
        3. **Encoding**:
           - Label Encoding untuk categorical features (County, Well Type Code)
        """)
        
        # Before and after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Sebelum Preprocessing")
            st.write("Missing Values:")
            missing_before = df.isnull().sum()
            st.dataframe(missing_before[missing_before > 0])
        
        with col2:
            st.markdown("#### ✅ Setelah Preprocessing")
            st.write("Missing Values:")
            missing_after = df_processed.isnull().sum()
            if missing_after.sum() == 0:
                st.success("✅ Semua missing values telah ditangani!")
            else:
                st.dataframe(missing_after[missing_after > 0])
        
        # Target distribution
        st.markdown("### 🎯 Distribusi Target Variable")
        target_dist = df_processed['Has_Production'].value_counts()
        fig = px.pie(values=target_dist.values, 
                    names=['No Production', 'Has Production'],
                    title='Distribution of Target Variable (Has_Production)')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🧠 Model Training":
        st.markdown('<h2 class="section-header">🧠 Pelatihan Model Decision Tree</h2>', unsafe_allow_html=True)
        
        # Train model
        with st.spinner("🔄 Training model..."):
            model, X_train, X_test, y_train, y_test, train_acc, test_acc, conf_matrix, class_report, county_encoder, well_type_encoder = train_model(df_processed)
        
        st.success("✅ Model berhasil dilatih!")
        
        # Model performance
        st.markdown("### 📊 Performa Model")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Training Accuracy", f"{train_acc:.3f}")
        with col2:
            st.metric("🎯 Testing Accuracy", f"{test_acc:.3f}")
        with col3:
            overfitting = train_acc - test_acc
            st.metric("⚠️ Overfitting Gap", f"{overfitting:.3f}")
        with col4:
            st.metric("🏆 Best Params", str(model.get_params()))
        
        # Confusion Matrix
        st.markdown("### 🔍 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Production', 'Has Production'],
                   yticklabels=['No Production', 'Has Production'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("### 📋 Classification Report")
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df)
    
    elif page == "🌳 Visualisasi Model":
        st.markdown('<h2 class="section-header">🌳 Visualisasi Model</h2>', unsafe_allow_html=True)
        
        # Train model for visualization
        model, X_train, X_test, y_train, y_test, train_acc, test_acc, conf_matrix, class_report, county_encoder, well_type_encoder = train_model(df_processed)
        
        # Feature Importance
        st.markdown("### 📊 Feature Importance")
        feature_names = ["County", "Well Type Code", "Months in Production", "Gas Produced, Mcf", 
                        "Water Produced, bbl", "Oil Produced, bbl", "Reporting Year"]
        feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        
        fig = px.bar(x=feature_importances.values, y=feature_importances.index,
                    orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
        
        # Decision Tree Visualization (simplified)
        st.markdown("### 🌳 Struktur Decision Tree")
        st.info("💡 Menampilkan struktur decision tree yang disederhanakan untuk interpretabilitas")
        
        # Create a simplified tree for visualization
        simple_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        simple_model.fit(X_train, y_train)
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(simple_model, filled=True, feature_names=feature_names, 
                 class_names=['No Production', 'Has Production'], rounded=True, fontsize=10)
        plt.title("Decision Tree Structure (Simplified - Max Depth 3)")
        st.pyplot(fig)
    
    elif page == "📈 Evaluasi Model":
        st.markdown('<h2 class="section-header">📈 Evaluasi Model</h2>', unsafe_allow_html=True)
        
        # Train model
        model, X_train, X_test, y_train, y_test, train_acc, test_acc, conf_matrix, class_report, county_encoder, well_type_encoder = train_model(df_processed)
        
        # Performance vs Max Depth
        st.markdown("### 📊 Performa vs Max Depth")
        
        depths = range(1, 21)
        train_scores = []
        test_scores = []
        
        for depth in depths:
            temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            temp_model.fit(X_train, y_train)
            train_scores.append(temp_model.score(X_train, y_train))
            test_scores.append(temp_model.score(X_test, y_test))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(depths), y=train_scores, mode='lines+markers', name='Training Accuracy'))
        fig.add_trace(go.Scatter(x=list(depths), y=test_scores, mode='lines+markers', name='Testing Accuracy'))
        fig.update_layout(title='Model Performance vs Max Depth',
                         xaxis_title='Max Depth',
                         yaxis_title='Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        
        # Overfitting Analysis
        st.markdown("### ⚠️ Analisis Overfitting/Underfitting")
        
        gap = train_acc - test_acc
        if gap < 0.05:
            st.success("✅ Model memiliki performa yang baik tanpa overfitting signifikan")
        elif gap < 0.1:
            st.warning("⚠️ Model menunjukkan sedikit overfitting")
        else:
            st.error("❌ Model mengalami overfitting yang signifikan")
        
        st.write(f"**Training Accuracy**: {train_acc:.3f}")
        st.write(f"**Testing Accuracy**: {test_acc:.3f}")
        st.write(f"**Gap**: {gap:.3f}")
    
    elif page == "📜 Rules Decision Tree":
        st.markdown('<h2 class="section-header">📜 Rules Decision Tree</h2>', unsafe_allow_html=True)
        
        # Train model
        model, X_train, X_test, y_train, y_test, train_acc, test_acc, conf_matrix, class_report, county_encoder, well_type_encoder = train_model(df_processed)
        
        st.markdown("### 📋 Aturan Decision Tree (Textual)")
        st.info("💡 Aturan ini menunjukkan bagaimana model membuat keputusan untuk memprediksi produksi")
        
        feature_names = ["County", "Well Type Code", "Months in Production", "Gas Produced, Mcf", 
                        "Water Produced, bbl", "Oil Produced, bbl", "Reporting Year"]
        
        # Create a simplified tree for better readability
        simple_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        simple_model.fit(X_train, y_train)
        
        tree_rules = export_text(simple_model, feature_names=feature_names)
        
        st.text(tree_rules)
        
        # Download option
        st.markdown("### 💾 Download")
        if st.button("📥 Download Decision Tree Rules"):
            st.download_button(
                label="Download Rules as Text File",
                data=tree_rules,
                file_name="decision_tree_rules.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()


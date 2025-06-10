# Import libraries
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
warnings.filterwarnings('ignore')

# Set style untuk visualisasi yang lebih menarik
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Semua library berhasil diimport!")

# Load dataset (pastikan file CSV sudah diupload ke Colab)
csv_file = 'oil-and-gas-annual-production-beginning-2001-1.csv'

# Baca data
df = pd.read_csv('oil-and-gas-annual-production-beginning-2001-1.csv')

print("="*60)
print("📊 INFORMASI DATASET")
print("="*60)
print(f"📋 Ukuran Dataset: {df.shape[0]:,} baris × {df.shape[1]} kolom")
print(f"📊 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print("\n📄 5 Baris Pertama:")
print(df.head())
print("\n📋 Info Dataset:")
print(df.info())

#Eksplorasi Data Awal
def data_exploration(df):
    """
    Eksplorasi data awal dengan visualisasi yang menarik
    """
    print("="*60)
    print("📊 EKSPLORASI DATA AWAL")
    print("="*60)

    # Missing values visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Missing values heatmap
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if len(missing_data) > 0:
        axes[0].barh(range(len(missing_data)), missing_data.values, color='coral')
        axes[0].set_yticks(range(len(missing_data)))
        axes[0].set_yticklabels(missing_data.index)
        axes[0].set_xlabel('Jumlah Missing Values')
        axes[0].set_title('🔍 Missing Values per Kolom', fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, '✅ Tidak ada Missing Values',
                    ha='center', va='center', transform=axes[0].transAxes, fontsize=14)
        axes[0].set_title('🔍 Status Missing Values', fontweight='bold')

    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
    wedges, texts, autotexts = axes[1].pie(dtype_counts.values, labels=dtype_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('📈 Distribusi Tipe Data', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Jalankan eksplorasi data
data_exploration(df)

# Preprocessing Data
def preprocessing(df):
    """
    Preprocessing data dengan visualisasi before/after
    """
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DATA")
    print("="*60)

    # Identifikasi target column
    target_col = 'Production_Flag' if 'Production_Flag' in df.columns else df.columns[-1]
    print(f"🎯 Target Column: {target_col}")

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
            df_clean[col] = le.fit_transform(df_clean[col].fillna('Unknown'))
            label_encoders[col] = le

    # After preprocessing stats
    after_shape = df_clean.shape
    after_missing = df_clean.isnull().sum().sum()

    # Visualization of preprocessing results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    categories = ['Jumlah Baris', 'Jumlah Kolom', 'Missing Values']
    before_values = [before_shape[0], before_shape[1], before_missing]
    after_values = [after_shape[0], after_shape[1], after_missing]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, before_values, width, label='Sebelum', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_values, width, label='Sesudah', color='lightblue', alpha=0.8)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Count')
    ax.set_title('📊 Perbandingan Before vs After Preprocessing', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print(f"✅ Preprocessing selesai:")
    print(f"   • Data shape: {before_shape} → {after_shape}")
    print(f"   • Missing values: {before_missing:,} → {after_missing:,}")

    return df_clean, target_col, label_encoders

# Jalankan preprocessing
df_clean, target_col, label_encoders = preprocessing(df)

#Split Data dan Persiapan Training

# Prepare features and target
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]
feature_names = X.columns.tolist()

print(f"📊 Feature columns: {len(feature_names)}")
print(f"🎯 Target column: {target_col}")
print(f"📈 Target distribution:")
print(y.value_counts().sort_index())

# Identify classes with only one sample
class_counts = y.value_counts()
single_instance_classes = class_counts[class_counts == 1].index

# Remove rows with single instance classes
df_filtered = df_clean[~df_clean[target_col].isin(single_instance_classes)].copy()

# Update features and target after filtering
X = df_filtered.drop(columns=[target_col])
y = df_filtered[target_col]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✅ Data split berhasil:")
print(f"   • Training set: {X_train.shape[0]:,} samples")
print(f"   • Testing set: {X_test.shape[0]:,} samples")

#Training Model dengan Hyperparameter Tuning
def train_model(X_train, X_test, y_train, y_test):
    """
    Training model dengan hyperparameter tuning
    """
    print("\n" + "="*60)
    print("🤖 TRAINING MODEL")
    print("="*60)

    # Hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    dt = DecisionTreeClassifier(random_state=42)

    print("🔍 Melakukan Grid Search untuk hyperparameter terbaik...")
    grid = GridSearchCV(
        dt, param_grid, cv=5, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )

    grid.fit(X_train, y_train)
    best_dt = grid.best_estimator_

    # Model evaluation
    train_score = best_dt.score(X_train, y_train)
    test_score = best_dt.score(X_test, y_test)

    print(f"✅ Training selesai!")
    print(f"   • Best parameters: {grid.best_params_}")
    print(f"   • Training accuracy: {train_score:.4f}")
    print(f"   • Testing accuracy: {test_score:.4f}")
    print(f"   • Cross-validation score: {grid.best_score_:.4f}")

    return best_dt

# Training model
best_dt = train_model(X_train, X_test, y_train, y_test)

#Confusion Matrix dan Classification Report
def visualize_confusion_matrix(best_dt, X_test, y_test):
    """
    Visualisasi confusion matrix yang lebih menarik
    """
    y_pred = best_dt.predict(X_test)

    # Get top 5 classes for cleaner visualization
    class_counts = pd.Series(y_test).value_counts()
    top_classes = class_counts.nlargest(5).index.tolist()

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=top_classes)

    # Create figure with custom styling
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create heatmap with better styling
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=[f'Class {c}' for c in top_classes],
        yticklabels=[f'Class {c}' for c in top_classes],
        cmap='Blues', cbar_kws={'label': 'Jumlah Prediksi'},
        square=True, linewidths=0.5, ax=ax
    )

    ax.set_title('🎯 Confusion Matrix (Top 5 Classes)', fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual Class', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Classification report
    print("📊 CLASSIFICATION REPORT (Top 5 Classes):")
    print("="*50)
    report = classification_report(y_test, y_pred, labels=top_classes)
    print(report)

# Jalankan visualisasi confusion matrix
visualize_confusion_matrix(best_dt, X_test, y_test)

#Visualisasi Decision Tree
def visualize_clean_decision_tree(best_dt, feature_names):
    """
    Visualisasi decision tree yang bersih dan jelas
    """
    print("\n" + "="*60)
    print("🌳 VISUALISASI DECISION TREE")
    print("="*60)

    # Create a large figure for clear tree visualization
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))

    # Plot the decision tree with optimal settings for clarity
    plot_tree(
        best_dt,
        max_depth=2,  # Increase depth for more detail but still readable
        filled=True,
        rounded=True,
        feature_names=feature_names,
        fontsize=12,  # Larger font for readability
        class_names=[f'Class_{c}' for c in best_dt.classes_],
        proportion=False,  # Show actual counts instead of proportions
        impurity=True,
        precision=2,
        ax=ax
    )

    ax.set_title('🌳 Decision Tree Structure (Max Depth = 4)',
                fontweight='bold', fontsize=20, pad=20)

    plt.tight_layout()
    plt.show()

    # Print tree statistics
    print(f"📊 Tree Statistics:")
    print(f"   • Total Depth: {best_dt.tree_.max_depth}")
    print(f"   • Number of Leaves: {best_dt.tree_.n_leaves}")
    print(f"   • Number of Nodes: {best_dt.tree_.node_count}")

def visualize_feature_importance_only(best_dt, feature_names):
    """
    Visualisasi feature importance terpisah dan lebih jelas
    """
    print("\n🏆 FEATURE IMPORTANCE ANALYSIS")
    print("="*50)

    # Feature importance analysis
    importances = pd.Series(best_dt.feature_importances_, index=feature_names)
    top_features = importances.nlargest(15)  # Show top 15 features

    # Create horizontal bar plot
    fig, ax = plt.subplots(1, 1, figsize=(2, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features.values, color=colors)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=11)
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
    ax.set_title('🏆 Top 15 Feature Importance', fontweight='bold', fontsize=16, pad=15)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

    # Print top 5 features
    print("🥇 TOP 5 MOST IMPORTANT FEATURES:")
    for i, (feature, importance) in enumerate(top_features.head().items(), 1):
        percentage = (importance / importances.sum()) * 100
        print(f"   {i}. {feature}: {importance:.4f} ({percentage:.2f}%)")

# Jalankan visualisasi yang diperbaiki
visualize_clean_decision_tree(best_dt, feature_names)
visualize_feature_importance_only(best_dt, feature_names)

def visualize_model_performance(best_dt, X_train, y_train, X_test, y_test):
    """
    Visualisasi performa model yang bersih dan fokus
    """
    print("\n📈 MODEL PERFORMANCE ANALYSIS")
    print("="*50)

    # Create a single focused visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Tree depth analysis
    depths = []
    train_scores = []
    test_scores = []

    for depth in range(1, 16):  # Extended range for better analysis
        temp_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        temp_tree.fit(X_train, y_train)
        train_score = temp_tree.score(X_train, y_train)
        test_score = temp_tree.score(X_test, y_test)

        depths.append(depth)
        train_scores.append(train_score)
        test_scores.append(test_score)

    # Plot both training and testing scores
    ax.plot(depths, train_scores, 'o-', linewidth=3, markersize=8,
            color='blue', label='Training Accuracy', alpha=0.8)
    ax.plot(depths, test_scores, 's-', linewidth=3, markersize=8,
            color='red', label='Testing Accuracy', alpha=0.8)

    ax.set_xlabel('Max Depth', fontweight='bold', fontsize=14)
    ax.set_ylabel('Accuracy Score', fontweight='bold', fontsize=14)
    ax.set_title('📈 Model Performance vs Tree Depth\n(Training vs Testing)',
                fontweight='bold', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')

    # Highlight best depth
    best_depth = best_dt.max_depth if best_dt.max_depth else 15
    if best_depth <= 15:
        ax.axvline(x=best_depth, color='green', linestyle='--', alpha=0.7,
                   linewidth=2, label=f'Selected Depth: {best_depth}')
        ax.legend(fontsize=12, loc='best')

    # Add annotations for best performance
    best_test_idx = np.argmax(test_scores)
    best_test_depth = depths[best_test_idx]
    best_test_score = test_scores[best_test_idx]

    ax.annotate(f'Best Test Accuracy: {best_test_score:.4f}\nAt Depth: {best_test_depth}',
                xy=(best_test_depth, best_test_score),
                xytext=(best_test_depth + 2, best_test_score - 0.05),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.show()

    # Print performance summary
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   • Best Testing Accuracy: {best_test_score:.4f} at depth {best_test_depth}")
    print(f"   • Current Model Accuracy: {best_dt.score(X_test, y_test):.4f}")
    print(f"   • Current Model Depth: {best_dt.tree_.max_depth}")
    print(f"   • Overfitting Check: {'⚠️ Possible overfitting' if train_scores[-1] - test_scores[-1] > 0.1 else '✅ Good generalization'}")

# Jalankan analisis performa
visualize_model_performance(best_dt, X_train, y_train, X_test, y_test)  

def show_tree_rules(best_dt, feature_names):
    """
    Menampilkan aturan decision tree dalam format teks
    """
    print("\n🔤 DECISION TREE RULES (Text Format):")
    print("="*50)
    tree_rules = export_text(
        best_dt,
        feature_names=feature_names,
        max_depth=4,  # Limit depth for readability
        spacing=3,
        decimals=2,
        show_weights=True
    )
    print(tree_rules[:2000] + "..." if len(tree_rules) > 2000 else tree_rules)

# Tampilkan tree rules
show_tree_rules(best_dt, feature_names)

def create_interactive_tree_graph(best_dt, feature_names):
    """
    Membuat visualisasi tree menggunakan Graphviz yang lebih menarik
    """
    print("\n🎨 Membuat visualisasi tree interaktif...")

    # Create GraphViz visualization
    dot_data = export_graphviz(
        best_dt,
        out_file=None,
        feature_names=feature_names,
        class_names=[f'Class_{c}' for c in best_dt.classes_],
        filled=True,
        rounded=True,
        max_depth=3,  # Limit for clarity
        proportion=True,
        impurity=True,
        precision=2,
        label='root'
    )

    # Create and display graph
    graph = graphviz.Source(dot_data)

    # Save as different formats
    try:
        graph.render("decision_tree_oil_gas", format="png", cleanup=True)
        print("✅ Tree saved as PNG file: decision_tree_oil_gas.png")
    except:
        print("⚠️ Could not save PNG file")

    return graph

# Buat interactive tree graph
graph = create_interactive_tree_graph(best_dt, feature_names)

# Display the graph
try:
    graph.view()  # This will open the graph in a viewer
except:
    print("Graph created but cannot display in Colab. Check files panel for PNG output.")

# Display in Colab
graph

def analyze_feature_distributions(df_clean, best_dt, feature_names):
    """
    Analisis distribusi fitur-fitur penting
    """
    print("\n" + "="*60)
    print("📈 ANALISIS DISTRIBUSI FITUR")
    print("="*60)

    # Get top features
    importances = pd.Series(best_dt.feature_importances_, index=feature_names)
    top_features = importances.nlargest(6).index.tolist()

    # Create subplots for distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('📊 Distribusi Top 6 Features Terpenting', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        # Check if feature is numeric or categorical
        if df_clean[feature].nunique() > 20:  # Numeric
            df_clean[feature].hist(bins=30, ax=axes[i], alpha=0.7, color=plt.cm.Set2(i))
            axes[i].axvline(df_clean[feature].mean(), color='red', linestyle='--', alpha=0.8,
                           label=f'Mean: {df_clean[feature].mean():.2f}')
        else:  # Categorical
            value_counts = df_clean[feature].value_counts().nlargest(10)
            value_counts.plot(kind='bar', ax=axes[i], color=plt.cm.Set2(i))
            axes[i].tick_params(axis='x', rotation=45)

        axes[i].set_title(f'{feature}\n(Importance: {importances[feature]:.3f})', fontweight='bold')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Jalankan analisis distribusi fitur
analyze_feature_distributions(df_clean, best_dt, feature_names)

def generate_business_insights(best_dt, feature_names, X_test, y_test):
    """
    Menghasilkan insights bisnis dari model
    """
    print("\n" + "="*60)
    print("💡 BUSINESS INSIGHTS")
    print("="*60)

    # Feature importance insights
    importances = pd.Series(best_dt.feature_importances_, index=feature_names)
    top_5_features = importances.nlargest(5)

    print("🏆 TOP 5 FAKTOR PENENTU PRODUKSI:")
    for i, (feature, importance) in enumerate(top_5_features.items(), 1):
        print(f"   {i}. {feature}: {importance:.3f} ({importance/importances.sum()*100:.1f}%)")

    # Model performance summary
    y_pred = best_dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 PERFORMA MODEL:")
    print(f"   • Akurasi: {accuracy:.2%}")
    print(f"   • Jumlah fitur digunakan: {len(feature_names)}")
    print(f"   • Kedalaman tree: {best_dt.tree_.max_depth}")
    print(f"   • Jumlah leaf nodes: {best_dt.tree_.n_leaves}")

    # Prediction distribution
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    print(f"\n🎯 DISTRIBUSI PREDIKSI:")
    for class_val, count in pred_dist.items():
        percentage = count / len(y_pred) * 100
        print(f"   • Class {class_val}: {count:,} sumur ({percentage:.1f}%)")


# Generate business insights
generate_business_insights(best_dt, feature_names, X_test, y_test)


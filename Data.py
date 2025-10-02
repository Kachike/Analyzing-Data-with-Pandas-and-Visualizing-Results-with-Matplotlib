# data_analysis_visualization.ipynb

# %% [markdown]
# # Data Analysis with Pandas and Visualization with Matplotlib
# 
# ## Analysis of the Iris Dataset
# 
# This notebook demonstrates:
# - Loading and exploring datasets with pandas
# - Basic data analysis and statistics
# - Creating various visualizations with matplotlib
# - Following best practices with error handling

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")

# %% [markdown]
# # Task 1: Load and Explore the Dataset

# %%
class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.dataset_name = None
    
    def load_iris_dataset(self):
        """Load the Iris dataset from sklearn and convert to pandas DataFrame"""
        try:
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
            self.dataset_name = "Iris Dataset"
            print("✅ Iris dataset loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def load_from_csv(self, file_path):
        """Load dataset from CSV file with error handling"""
        try:
            self.df = pd.read_csv(file_path)
            self.dataset_name = file_path
            print(f"✅ Dataset loaded successfully from {file_path}")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File {file_path} not found.")
            return False
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def explore_dataset(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("❌ No dataset loaded!")
            return
        
        print("=" * 50)
        print(f"EXPLORING: {self.dataset_name}")
        print("=" * 50)
        
        # Display first few rows
        print("\n📊 First 10 rows of the dataset:")
        display(self.df.head(10))
        
        # Dataset shape
        print(f"\n📐 Dataset Shape: {self.df.shape}")
        
        # Data types and missing values
        print("\n🔍 Data Types and Missing Values:")
        info_df = pd.DataFrame({
            'Data Type': self.df.dtypes,
            'Missing Values': self.df.isnull().sum(),
            'Missing %': (self.df.isnull().sum() / len(self.df)) * 100
        })
        display(info_df)
        
        # Column information
        print(f"\n📋 Columns: {list(self.df.columns)}")
    
    def clean_dataset(self):
        """Clean the dataset by handling missing values"""
        if self.df is None:
            print("❌ No dataset loaded!")
            return
        
        print("\n🧹 Cleaning Dataset...")
        
        # Check for missing values
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        if missing_before > 0:
            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = self.df[numerical_cols].fillna(self.df[numerical_cols].median())
            
            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
            missing_after = self.df.isnull().sum().sum()
            print(f"Missing values after cleaning: {missing_after}")
        else:
            print("✅ No missing values found!")

# %%
# Initialize the analyzer and load data
analyzer = DataAnalyzer()

# Load the Iris dataset
if not analyzer.load_iris_dataset():
    # Fallback: Try to load from CSV if sklearn fails
    print("Trying to load from CSV...")
    analyzer.load_from_csv("iris.csv")

# Explore the dataset
analyzer.explore_dataset()

# Clean the dataset
analyzer.clean_dataset()

# %% [markdown]
# # Task 2: Basic Data Analysis

# %%
class DataAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def basic_statistics(self):
        """Compute basic statistics for numerical columns"""
        print("=" * 50)
        print("📈 BASIC STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Basic describe() method
        print("\n📊 Overall Statistics (describe()):")
        display(self.df.describe())
        
        # Additional statistics
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\n📋 Numerical columns: {list(numerical_cols)}")
        
        # Mean, median, std for each numerical column
        stats_df = pd.DataFrame({
            'Mean': self.df[numerical_cols].mean(),
            'Median': self.df[numerical_cols].median(),
            'Std Dev': self.df[numerical_cols].std(),
            'Variance': self.df[numerical_cols].var(),
            'Min': self.df[numerical_cols].min(),
            'Max': self.df[numerical_cols].max()
        })
        
        print("\n📊 Detailed Statistics:")
        display(stats_df.round(3))
    
    def group_analysis(self, group_column, value_column):
        """Perform grouping analysis on categorical columns"""
        print(f"\n🔍 GROUP ANALYSIS: {value_column} by {group_column}")
        print("-" * 40)
        
        if group_column not in self.df.columns or value_column not in self.df.columns:
            print("❌ Specified columns not found in dataset!")
            return
        
        try:
            grouped_stats = self.df.groupby(group_column)[value_column].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(3)
            
            print(f"📊 {value_column} statistics by {group_column}:")
            display(grouped_stats)
            
            # Additional insights
            print(f"\n💡 INSIGHTS for {value_column} by {group_column}:")
            max_group = grouped_stats['mean'].idxmax()
            min_group = grouped_stats['mean'].idxmin()
            print(f"• Highest average {value_column}: {max_group} ({grouped_stats.loc[max_group, 'mean']})")
            print(f"• Lowest average {value_column}: {min_group} ({grouped_stats.loc[min_group, 'mean']})")
            print(f"• Variability (std dev range): {grouped_stats['std'].min():.3f} to {grouped_stats['std'].max():.3f}")
            
        except Exception as e:
            print(f"❌ Error in group analysis: {e}")
    
    def correlation_analysis(self):
        """Perform correlation analysis on numerical columns"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("❌ Not enough numerical columns for correlation analysis!")
            return
        
        print("\n📊 CORRELATION MATRIX")
        print("-" * 30)
        correlation_matrix = self.df[numerical_cols].corr().round(3)
        display(correlation_matrix)
        
        # Find strongest correlations
        print("\n💡 STRONGEST CORRELATIONS:")
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                corr_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation strength
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for col1, col2, corr in corr_pairs[:3]:  # Top 3
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            direction = "positive" if corr > 0 else "negative"
            print(f"• {col1} vs {col2}: {corr} ({strength} {direction} correlation)")

# %%
# Perform basic data analysis
analysis = DataAnalysis(analyzer.df)

# Basic statistics
analysis.basic_statistics()

# Group analysis by species
analysis.group_analysis('species', 'sepal length (cm)')
analysis.group_analysis('species', 'petal length (cm)')

# Correlation analysis
analysis.correlation_analysis()

# %% [markdown]
# # Task 3: Data Visualization

# %%
class DataVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup consistent plot style"""
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
    
    def create_line_chart(self, x_col, y_col, title="Line Chart"):
        """Create a line chart showing trends"""
        try:
            plt.figure(figsize=(12, 6))
            
            # If we have a time series, use it directly
            # For Iris dataset

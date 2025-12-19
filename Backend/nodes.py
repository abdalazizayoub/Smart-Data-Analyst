import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
from typing import Tuple, Union, Literal
import seaborn as sns

df = pd.read_csv(r"c:\Users\user\Downloads\archive(1)\house_price_regression_dataset.csv")


  
# Function to generate dataset description
def get_dataset_description(dataset_name ,data):
    try:
        df = pd.read_csv(BytesIO(data))
        columns = [str(col) for col in df.columns]
        
        # Separate numeric and non-numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Generate summary for numeric columns only
        numeric_summary = {}
        if numeric_columns:
            numeric_df = df[numeric_columns]
            summary_df = numeric_df.describe()
            
            for column in summary_df.columns:
                numeric_summary[str(column)] = {
                    str(stat): float(value) if pd.notna(value) else None
                    for stat, value in summary_df[column].items()
                }
        
        text_summary = {}
        for col in text_columns:
            text_summary[str(col)] = {
                "count": int(df[col].count()),
                "unique_values_list": df[col].dropna().unique().tolist(),
                "top_value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,  
                                }
        
        return {
            "Dataset Name": dataset_name,
            "columns": columns,
            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
            "numeric_summary": numeric_summary,
            "text_summary": text_summary,
            "dataset_shape": {
                "rows": int(len(df)),
                "columns": int(len(df.columns))
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to generate description: {str(e)}"}
#removes any row with NA data
def drop_na_rows(df:pd.DataFrame):
    df = df.copy()
    df.dropna(inplace=True)

    return df 

#Computes correlation matrix and return the matrix and a heatmap 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Union, Optional
import warnings
warnings.filterwarnings('ignore')


def correlation_analysis(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    heatmap_color: str = "rocket",
    figsize: Tuple[int, int] = (12, 10),
    annotation: bool = True,
    significance_level: float = 0.05
) -> Union[Dict, Tuple]:
    """
    Perform correlation analysis with dynamic handling for regression and classification tasks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, optional
        Name of the target column. If provided, computes correlations with target
    task_type : str, optional
        'classification' or 'regression'. If None, auto-detects based on target dtype
    heatmap_color : str, default='rocket'
        Color palette for heatmap. Options: 'rocket', 'mako', 'flare', 'crest', 'viridis', 'coolwarm'
    figsize : tuple, default=(12, 10)
        Figure size for plots
    annotation : bool, default=True
        Whether to annotate heatmap with correlation values
    significance_level : float, default=0.05
        Significance level for statistical tests
        
    Returns:
    --------
    Dictionary containing:
    - 'correlation_matrix': Full correlation matrix
    - 'target_correlations': Correlations with target (if target_column provided)
    - 'significant_features': Features significantly correlated with target
    - 'heatmap': Heatmap figure
    - 'task_type': Detected/used task type
    - 'test_used': Statistical test used
    """
    
    try:
        df = df.copy()
        
        # Available color palettes
        valid_colors = ["rocket", "mako", "flare", "crest", "viridis", "coolwarm", "Spectral", "RdYlBu"]
        if heatmap_color not in valid_colors:
            return {"error": f"Heatmap color must be one of {valid_colors}"}
        
        # If no target column provided, just do feature-feature correlation
        if target_column is None:
            return _feature_feature_correlation(df, heatmap_color, figsize, annotation)
        
        # Determine task type if not provided
        if task_type is None:
            task_type = _detect_task_type(df[target_column])
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if task_type == 'regression':
            return _regression_correlation_analysis(
                X, y, target_column, heatmap_color, figsize, annotation, significance_level
            )
        else:  # classification
            return _classification_correlation_analysis(
                X, y, target_column, heatmap_color, figsize, annotation, significance_level
            )
        
    except Exception as e:
        return {"error": f"Error while computing correlation analysis: {str(e)}"}


def _detect_task_type(target_series: pd.Series) -> str:
    """Auto-detect if task is classification or regression based on target column."""
    # Check if it's categorical/object type
    if target_series.dtype == 'object' or pd.api.types.is_string_dtype(target_series):
        unique_vals = target_series.nunique()
        # If few unique values relative to dataset size, likely classification
        if unique_vals < 0.1 * len(target_series) or unique_vals < 10:
            return 'classification'
        else:
            # Try to convert to numeric for regression
            try:
                pd.to_numeric(target_series)
                return 'regression'
            except:
                return 'classification'
    
    # Check if numeric type
    elif pd.api.types.is_numeric_dtype(target_series):
        unique_vals = target_series.nunique()
        # If integer with few unique values, likely classification
        if unique_vals < 10 and pd.api.types.is_integer_dtype(target_series):
            return 'classification'
        else:
            return 'regression'
    
    # Check if categorical type
    elif pd.api.types.is_categorical_dtype(target_series):
        return 'classification'
    
    # Default to regression for numeric-like data
    else:
        return 'regression'


def _feature_feature_correlation(
    df: pd.DataFrame, 
    heatmap_color: str, 
    figsize: Tuple[int, int], 
    annotation: bool
) -> Dict:
    """Handle correlation analysis when no target column is specified."""
    # Convert categorical columns to numeric for correlation
    df_encoded = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            if df[col].nunique() <= 10:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
            else:
                # Use frequency encoding for high cardinality
                freq = df[col].value_counts(normalize=True)
                df_encoded[col] = df[col].map(freq)
    
    # Compute correlation matrix
    corr_matrix = df_encoded.corr(method='pearson')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap=heatmap_color,
        annot=annotation,
        fmt='.2f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=16, pad=20)
    plt.tight_layout()
    
    return {
        'correlation_matrix': corr_matrix,
        'heatmap': fig,
        'task_type': 'feature_analysis',
        'test_used': 'pearson'
    }


def _regression_correlation_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    heatmap_color: str,
    figsize: Tuple[int, int],
    annotation: bool,
    significance_level: float
) -> Dict:
    """Perform correlation analysis for regression tasks using Pearson correlation."""
    # Prepare data for correlation
    data = X.copy()
    
    # Convert categorical features to numeric
    for col in X.columns:
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            if X[col].nunique() <= 10:
                le = LabelEncoder()
                data[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
            else:
                # Use mean encoding based on target
                encoding_dict = y.groupby(X[col]).mean().to_dict()
                data[col] = X[col].map(encoding_dict).fillna(y.mean())
        elif pd.api.types.is_categorical_dtype(X[col]):
            data[col] = X[col].cat.codes
    
    # Add target column
    data[target_column] = y
    
    # Compute full correlation matrix
    corr_matrix = data.corr(method='pearson')
    
    # Extract correlations with target
    target_correlations = corr_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
    
    # Compute p-values for significance
    n = len(data)
    significant_features = []
    for feature in X.columns:
        if feature in data.columns:
            corr_value = corr_matrix.loc[feature, target_column]
            if not pd.isna(corr_value):
                # t-test for Pearson correlation
                t_stat = corr_value * np.sqrt(n - 2) / np.sqrt(1 - corr_value**2)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                if p_value < significance_level:
                    significant_features.append({
                        'feature': feature,
                        'correlation': corr_value,
                        'p_value': p_value,
                        'significant': True
                    })
    
    # Create two subplots
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//1.5))
    
    # Plot 1: Full correlation matrix
    sns.heatmap(
        corr_matrix,
        cmap=heatmap_color,
        annot=annotation,
        fmt='.2f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=axes[0]
    )
    axes[0].set_title(f'Full Correlation Matrix\n(Pearson)', fontsize=14)
    
    # Plot 2: Top correlations with target
    top_n = min(15, len(target_correlations))
    top_features = target_correlations.head(top_n)
    
    colors = ['red' if x < 0 else 'blue' for x in top_features.values]
    axes[1].barh(range(len(top_features)), top_features.values, color=colors)
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].set_title(f'Top {top_n} Features Correlated with {target_column}', fontsize=14)
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add correlation values on bars
    for i, (idx, val) in enumerate(top_features.items()):
        axes[1].text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                    va='center', ha='left' if val >= 0 else 'right',
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    return {
        'correlation_matrix': corr_matrix,
        'target_correlations': target_correlations,
        'significant_features': pd.DataFrame(significant_features),
        'heatmap': fig,
        'task_type': 'regression',
        'test_used': 'pearson',
        'significance_level': significance_level
    }


def _classification_correlation_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    heatmap_color: str,
    figsize: Tuple[int, int],
    annotation: bool,
    significance_level: float
) -> Dict:
    """Perform correlation analysis for classification tasks using Chi-square test."""
    # Encode target if categorical
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y) or pd.api.types.is_categorical_dtype(y):
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        target_classes = le_target.classes_
    else:
        y_encoded = y.values
        target_classes = np.unique(y_encoded)
    
    # Prepare data for analysis
    chi2_results = []
    cramers_v_matrix = pd.DataFrame(index=X.columns, columns=['chi2_stat', 'p_value', 'cramers_v'])
    
    # Compute Chi-square for each feature
    for col in X.columns:
        # Prepare contingency table
        if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
            # For categorical features, use actual values
            feature_series = X[col].astype(str).fillna('missing')
            contingency = pd.crosstab(feature_series, y_encoded)
        elif pd.api.types.is_categorical_dtype(X[col]):
            # For categorical dtype
            contingency = pd.crosstab(X[col], y_encoded)
        else:
            # For numerical features, bin them
            feature_series = pd.qcut(X[col].rank(method='first'), q=5, duplicates='drop')
            contingency = pd.crosstab(feature_series, y_encoded)
        
        # Perform Chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        # Compute Cramér's V (effect size)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        chi2_results.append({
            'feature': col,
            'chi2_statistic': chi2,
            'p_value': p,
            'cramers_v': cramers_v,
            'degrees_of_freedom': dof,
            'significant': p < significance_level
        })
    
    chi2_df = pd.DataFrame(chi2_results)
    chi2_df = chi2_df.sort_values('cramers_v', ascending=False)
    
    # Create Cramér's V correlation-like matrix
    # For simplicity, we'll create a matrix with Cramér's V values
    cramers_v_series = chi2_df.set_index('feature')['cramers_v']
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//1.5))
    
    # Plot 1: Top features by Cramér's V
    top_n = min(15, len(cramers_v_series))
    top_features = cramers_v_series.head(top_n)
    
    axes[0].barh(range(len(top_features)), top_features.values, color='purple', alpha=0.7)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Cramér's V")
    axes[0].set_title(f'Top {top_n} Features (Chi-square)\nvs {target_column}', fontsize=14)
    
    # Add values on bars
    for i, (idx, val) in enumerate(top_features.items()):
        axes[0].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: P-value distribution
    significant_count = chi2_df['significant'].sum()
    total_count = len(chi2_df)
    
    labels = ['Significant', 'Not Significant']
    sizes = [significant_count, total_count - significant_count]
    colors = ['lightgreen', 'lightcoral']
    
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].axis('equal')
    axes[1].set_title(f'Feature Significance\n(α = {significance_level})', fontsize=14)
    
    plt.tight_layout()
    
    return {
        'chi2_results': chi2_df,
        'cramers_v_values': cramers_v_series,
        'significant_features': chi2_df[chi2_df['significant']],
        'heatmap': fig,
        'task_type': 'classification',
        'test_used': 'chi_square',
        'significance_level': significance_level,
        'target_classes': target_classes.tolist() if hasattr(target_classes, 'tolist') else list(target_classes)
    }


# Simplified wrapper function
def correlation_analysis_simple(
    df: pd.DataFrame,
    target_column: str = None,
    heatmap_color: str = "rocket"
) -> Union[Dict, Tuple]:
    """Simplified version for backward compatibility."""
    return correlation_analysis(
        df=df,
        target_column=target_column,
        heatmap_color=heatmap_color,
        task_type=None,  # Auto-detect
        figsize=(12, 10),
        annotation=True,
        significance_level=0.05
    )


# Example usage function
def analyze_correlations(df, target_col=None, task_type=None):
    """
    Example function demonstrating how to use the correlation analysis.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    target_col : str, optional
        Target column name
    task_type : str, optional
        'classification' or 'regression'
    """
    
    print(f"Analyzing correlations for dataset with {len(df)} rows and {len(df.columns)} columns")
    if target_col:
        print(f"Target column: {target_col}")
    
    results = correlation_analysis(
        df=df,
        target_column=target_col,
        task_type=task_type,
        heatmap_color='coolwarm',
        figsize=(14, 10),
        annotation=True
    )
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\nTask Type: {results['task_type'].upper()}")
    print(f"Statistical Test Used: {results['test_used'].upper()}")
    
    if target_col and results['task_type'] == 'regression':
        print(f"\nTop 5 correlations with '{target_col}':")
        print(results['target_correlations'].head())
        
        if 'significant_features' in results and not results['significant_features'].empty:
            print(f"\nSignificant features (α = {results.get('significance_level', 0.05)}):")
            print(f"Found {len(results['significant_features'])} significant features")
    
    elif target_col and results['task_type'] == 'classification':
        print(f"\nTop 5 features by Cramér's V:")
        print(results['cramers_v_values'].head())
        
        if 'significant_features' in results and not results['significant_features'].empty:
            print(f"\nSignificant features (α = {results.get('significance_level', 0.05)}):")
            print(f"Found {len(results['significant_features'])} significant features")
    
    # Display the plot
    plt.show()
    
    return results
    
    
    
    

# Computes Feature importance and returns feature importnace, bar plot , test score of the rf    

def feature_importance(
    df: pd.DataFrame, 
    class_label: str, 
    test_size: float = 0.2,
    task_type: str = None  # 'classification' or 'regression'
) -> Union[Tuple, dict]:
    try:
        # Create a copy to avoid modifying original dataframe
        df = df.copy()
        
        # Determine task type if not provided
        if task_type is None:
            # Auto-detect based on target dtype
            if df[class_label].dtype == 'object' or pd.api.types.is_string_dtype(df[class_label]):
                # Check if string column contains categorical data
                unique_values = df[class_label].nunique()
                if unique_values < 0.1 * len(df):  # Heuristic: if few unique values relative to dataset size
                    task_type = 'classification'
                else:
                    # For string columns with many unique values, check if they can be converted to numeric
                    try:
                        pd.to_numeric(df[class_label])
                        task_type = 'regression'
                    except:
                        task_type = 'classification'  # Default to classification
            elif pd.api.types.is_numeric_dtype(df[class_label]):
                # Check if numeric column looks like classification (integer with few unique values)
                unique_values = df[class_label].nunique()
                if unique_values < 10 and df[class_label].dtype in ['int64', 'int32']:
                    task_type = 'classification'
                else:
                    task_type = 'regression'
            else:
                task_type = 'classification'  # Default
        
        # Prepare features and target
        X = df.drop(columns=[class_label])
        y = df[class_label]
        
        feature_names = X.columns.tolist()
        
        # Handle encoding for classification tasks
        if task_type == 'classification':
            X_encoded = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                    if X[col].nunique() <= 10:  
                        encoder = LabelEncoder()
                        X_encoded[col] = encoder.fit_transform(X[col].astype(str).fillna('missing'))
                    else: 
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_encoded[col] = encoder.fit_transform(X[col].astype(str).fillna('missing').values.reshape(-1, 1))
                elif pd.api.types.is_categorical_dtype(X[col]):
                    X_encoded[col] = X[col].cat.codes
            
            if y.dtype == 'object' or pd.api.types.is_string_dtype(y) or pd.api.types.is_categorical_dtype(y):
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y.astype(str))
            else:
                y_encoded = y.values
                
            X_final = X_encoded.values
            y_final = y_encoded
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            
        else:  # Regression task
            X_encoded = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                    # For regression, use ordinal encoding for categorical features
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X_encoded[col] = encoder.fit_transform(X[col].astype(str).fillna('missing').values.reshape(-1, 1))
                elif pd.api.types.is_categorical_dtype(X[col]):
                    X_encoded[col] = X[col].cat.codes
            
            X_final = X_encoded.values
            y_final = y.values
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, test_size=test_size, random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        importance = model.feature_importances_
        
        # Create importance series
        importance_series = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_series)))
        bars = ax.barh(range(len(importance_series)), importance_series.values, color=colors)
        ax.set_yticks(range(len(importance_series)))
        ax.set_yticklabels(importance_series.index)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance ({task_type.capitalize()})\nTest Score: {test_score:.4f}')
        
        # Add value labels on bars
        for i, (value, bar) in enumerate(zip(importance_series.values, bars)):
            ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Return results
        results = {
            'importance': importance_series,
            'plot': fig,
            'test_score': test_score,
            'task_type': task_type,
            'model': model
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Error while computing feature importance: {str(e)}"}


def process_dataset(df, target_column, task_type=None):
    """
    Helper function to demonstrate usage
    """
    results = feature_importance(
        df=df,
        class_label=target_column,
        test_size=0.2,
        task_type=task_type  # Can be 'classification', 'regression'
    )
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Task Type: {results['task_type']}")
    print(f"Test Score: {results['test_score']:.4f}")
    print("\nTop 10 Features:")
    print(results['importance'].head(10))
    
    plt.show()
    
    return results


        
test_1 = drop_na_rows(df=df)
corr,_= correlation_analysis(df=df,heatmap_color="crest")
imp = feature_importance(df,class_label="House_Price",test_size=0.2)

print(f"removing null rows:\n {df.shape} after removing {test_1.shape}")
print("-"*50)
print(f"heatmap:{_.show()}")
print("-"*50)
print(f"feature_importance :{imp}")
print(test_1["House_Price"].dtypes)


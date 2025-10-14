"""
Evaluation and visualization framework for air quality forecasting models.
Provides comprehensive metrics and visualization tools for model assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Model evaluation class for air quality forecasting.
    Provides comprehensive evaluation metrics and visualization tools.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize ModelEvaluator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.get('logging', {}).get('level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                self.config.get('logging', {}).get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            self.logger.warning(f"No valid predictions for {model_name}")
            return metrics
        
        try:
            # Basic regression metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
            metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
            metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
            metrics['explained_variance'] = explained_variance_score(y_true_clean, y_pred_clean)
            
            # MAPE (handle division by zero)
            mape_mask = y_true_clean != 0
            if np.any(mape_mask):
                metrics['mape'] = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) 
                                               / y_true_clean[mape_mask])) * 100
            else:
                metrics['mape'] = np.nan
            
            # Bias and normalized metrics
            metrics['bias'] = np.mean(y_pred_clean - y_true_clean)
            metrics['normalized_rmse'] = metrics['rmse'] / np.mean(y_true_clean) * 100
            metrics['normalized_mae'] = metrics['mae'] / np.mean(y_true_clean) * 100
            
            # Correlation metrics
            correlation, p_value = pearsonr(y_true_clean, y_pred_clean)
            metrics['correlation'] = correlation
            metrics['correlation_p_value'] = p_value
            
            # Spearman rank correlation
            spearman_corr, spearman_p = spearmanr(y_true_clean, y_pred_clean)
            metrics['spearman_correlation'] = spearman_corr
            metrics['spearman_p_value'] = spearman_p
            
            # Additional metrics for air quality
            metrics['accuracy_10percent'] = np.mean(np.abs(y_pred_clean - y_true_clean) / y_true_clean <= 0.1) * 100
            metrics['accuracy_20percent'] = np.mean(np.abs(y_pred_clean - y_true_clean) / y_true_clean <= 0.2) * 100
            
            self.logger.info(f"Calculated metrics for {model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {model_name}: {e}")
        
        return metrics
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "Model", is_sequence: bool = False,
                      sequence_length: int = 24) -> Dict[str, float]:
        """
        Evaluate a single model and return metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            is_sequence: Whether the model requires sequence input
            sequence_length: Length of sequences for LSTM models
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if is_sequence:
                # Prepare sequence data for LSTM/CNN-LSTM models
                X_test_seq, y_test_seq = self._prepare_sequence_data(X_test, y_test, sequence_length)
                y_pred = model.predict(X_test_seq).flatten()
                y_true = y_test_seq
            else:
                # Standard prediction for ML models
                y_pred = model.predict(X_test)
                y_true = y_test
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            return {}
    
    def _prepare_sequence_data(self, X: np.ndarray, y: np.ndarray, 
                              sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM models."""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def compare_models(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray,
                      target_name: str = "Target") -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            target_name: Name of the target variable
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Determine if model requires sequence input
            is_sequence = 'lstm' in model_name.lower()
            
            # Evaluate model
            metrics = self.evaluate_model(
                model, X_test, y_test, model_name, 
                is_sequence=is_sequence
            )
            
            if metrics:
                metrics['model_name'] = model_name
                metrics['target'] = target_name
                comparison_results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            # Sort by R² score (descending)
            comparison_df = comparison_df.sort_values('r2', ascending=False).reset_index(drop=True)
            self.logger.info(f"Model comparison completed for {len(comparison_df)} models")
        
        return comparison_df
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str = "Model", 
                                  target_name: str = "Target",
                                  save_path: str = None) -> None:
        """
        Create scatter plot of predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            target_name: Name of the target variable
            save_path: Path to save the plot
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            self.logger.warning(f"No valid data for plotting {model_name}")
            return
        
        # Calculate metrics for plot title
        r2 = r2_score(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(np.min(y_true_clean), np.min(y_pred_clean))
        max_val = max(np.max(y_true_clean), np.max(y_pred_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(y_true_clean, y_pred_clean, 1)
        p = np.poly1d(z)
        ax.plot(y_true_clean, p(y_true_clean), "g-", alpha=0.8, lw=2, label=f'Trend Line (slope={z[0]:.2f})')
        
        ax.set_xlabel(f'Actual {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{model_name}: Predictions vs Actual\\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = "Model", target_name: str = "Target",
                      save_path: str = None) -> None:
        """
        Create residual plots for model diagnosis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            target_name: Name of the target variable
            save_path: Path to save the plot
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            self.logger.warning(f"No valid data for plotting {model_name}")
            return
        
        residuals = y_true_clean - y_pred_clean
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred_clean, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[0, 1].scatter(y_true_clean, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name}: Residual Analysis for {target_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved residual plot to {save_path}")
        
        plt.show()
    
    def plot_time_series_forecast(self, dates: np.ndarray, y_true: np.ndarray, 
                                 y_pred: np.ndarray, model_name: str = "Model",
                                 target_name: str = "Target", save_path: str = None) -> None:
        """
        Create time series plot comparing predictions with actual values.
        
        Args:
            dates: Array of datetime values
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            target_name: Name of the target variable
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Plot actual and predicted values
        ax.plot(dates, y_true, label='Actual', linewidth=2, alpha=0.8)
        ax.plot(dates, y_pred, label='Predicted', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(target_name)
        ax.set_title(f'{model_name}: Time Series Forecast for {target_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved time series plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               model_name: str = "Model", top_k: int = 20,
                               save_path: str = None) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_k: Number of top features to show
            save_path: Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning(f"Model {model_name} does not have feature importances")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame and sort
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top K features
        top_features = feature_importance_df.head(top_k)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name}: Top {top_k} Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Invert y-axis to show most important at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def create_interactive_comparison_plot(self, comparison_df: pd.DataFrame,
                                         metric: str = 'r2', save_path: str = None) -> None:
        """
        Create interactive comparison plot using Plotly.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to display
            save_path: Path to save the HTML plot
        """
        if comparison_df.empty:
            self.logger.warning("No data to plot")
            return
        
        # Create bar plot
        fig = px.bar(
            comparison_df.sort_values(metric, ascending=False),
            x='model_name',
            y=metric,
            title=f'Model Comparison: {metric.upper()}',
            labels={'model_name': 'Model', metric: metric.upper()},
            color=metric,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved interactive plot to {save_path}")
        
        fig.show()
    
    def create_comprehensive_report(self, comparison_df: pd.DataFrame, 
                                   target_name: str = "Target",
                                   save_path: str = None) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            comparison_df: DataFrame with model comparison results
            target_name: Name of the target variable
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if comparison_df.empty:
            return "No model results to report."
        
        report = []
        report.append(f"# Air Quality Forecasting Model Evaluation Report")
        report.append(f"## Target Variable: {target_name}")
        report.append(f"## Number of Models Evaluated: {len(comparison_df)}")
        report.append("")
        
        # Best model summary
        best_model = comparison_df.iloc[0]
        report.append(f"## Best Performing Model")
        report.append(f"**Model:** {best_model['model_name']}")
        report.append(f"- **R²:** {best_model['r2']:.4f}")
        report.append(f"- **RMSE:** {best_model['rmse']:.4f}")
        report.append(f"- **MAE:** {best_model['mae']:.4f}")
        if 'mape' in best_model and not np.isnan(best_model['mape']):
            report.append(f"- **MAPE:** {best_model['mape']:.2f}%")
        report.append(f"- **Correlation:** {best_model['correlation']:.4f}")
        report.append("")
        
        # Model rankings
        report.append("## Model Rankings (by R²)")
        report.append("| Rank | Model | R² | RMSE | MAE | Correlation |")
        report.append("|------|-------|----|----- |----- |-------------|")
        
        for i, row in comparison_df.iterrows():
            report.append(f"| {i+1} | {row['model_name']} | {row['r2']:.4f} | "
                         f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['correlation']:.4f} |")
        
        report.append("")
        
        # Performance analysis
        report.append("## Performance Analysis")
        
        # R² analysis
        r2_mean = comparison_df['r2'].mean()
        r2_std = comparison_df['r2'].std()
        report.append(f"- **Average R²:** {r2_mean:.4f} ± {r2_std:.4f}")
        
        # RMSE analysis
        rmse_mean = comparison_df['rmse'].mean()
        rmse_std = comparison_df['rmse'].std()
        report.append(f"- **Average RMSE:** {rmse_mean:.4f} ± {rmse_std:.4f}")
        
        # Model type analysis
        if any('lstm' in name.lower() for name in comparison_df['model_name']):
            deep_learning_models = comparison_df[comparison_df['model_name'].str.contains('lstm', case=False)]
            traditional_models = comparison_df[~comparison_df['model_name'].str.contains('lstm', case=False)]
            
            if not deep_learning_models.empty and not traditional_models.empty:
                dl_avg_r2 = deep_learning_models['r2'].mean()
                trad_avg_r2 = traditional_models['r2'].mean()
                report.append(f"- **Deep Learning Models Avg R²:** {dl_avg_r2:.4f}")
                report.append(f"- **Traditional ML Models Avg R²:** {trad_avg_r2:.4f}")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if best_model['r2'] > 0.8:
            report.append("- The best model shows excellent predictive performance (R² > 0.8)")
        elif best_model['r2'] > 0.6:
            report.append("- The best model shows good predictive performance (R² > 0.6)")
        else:
            report.append("- Consider feature engineering or ensemble methods to improve performance")
        
        if best_model['correlation'] > 0.9:
            report.append("- Strong correlation indicates good model fit")
        
        # Combine report
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Saved evaluation report to {save_path}")
        
        return report_text
    
    def save_metrics_to_csv(self, comparison_df: pd.DataFrame, save_path: str):
        """
        Save comparison metrics to CSV file.
        
        Args:
            comparison_df: DataFrame with model comparison results
            save_path: Path to save the CSV file
        """
        try:
            comparison_df.to_csv(save_path, index=False)
            self.logger.info(f"Saved metrics to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics to CSV: {e}")
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "Model", save_path: str = None):
        """
        Plot the distribution of prediction errors.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of errors
        axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
        axes[0].axvline(x=np.mean(errors), color='g', linestyle='-', label=f'Mean Error: {np.mean(errors):.4f}')
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name}: Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot of errors
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel('Prediction Error')
        axes[1].set_title(f'{model_name}: Error Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved error distribution plot to {save_path}")
        
        plt.show()

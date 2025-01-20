import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class StudentPerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        """
        Preprocess the input data by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        # Create a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
        # Fill categorical missing values with mode
        for col in categorical_columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # Encode categorical variables
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        return df_processed
    
    def prepare_features_target(self, df, target_column):
        """
        Separate features and target variable, scale features.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """
        Train the random forest model and perform cross-validation.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring='r2'
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_mse': mse,
            'test_r2': r2,
            'test_rmse': np.sqrt(mse),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def plot_feature_importance(self, feature_importance):
        """
        Create a visualization of feature importance.
        """
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.title('Feature Importance in Student Performance Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        return plt

    def predict(self, X):
        """
        Make predictions on new data.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# Example usage
def main():
    # Sample data structure (replace with actual data)
    data = pd.DataFrame({
        'attendance_rate': np.random.uniform(60, 100, 1000),
        'study_hours': np.random.uniform(0, 8, 1000),
        'previous_grades': np.random.uniform(50, 100, 1000),
        'extracurricular': np.random.choice(['Yes', 'No'], 1000),
        'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'sleep_hours': np.random.uniform(4, 10, 1000),
        'final_grade': np.random.uniform(50, 100, 1000)
    })
    
    # Initialize predictor
    predictor = StudentPerformancePredictor()
    
    # Preprocess data
    processed_data = predictor.preprocess_data(data)
    
    # Prepare features and target
    X, y = predictor.prepare_features_target(processed_data, 'final_grade')
    
    # Train and evaluate model
    results = predictor.train_model(X, y)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Cross-validation R² scores: {results['cv_scores']}")
    print(f"Mean CV R² score: {results['cv_mean']:.3f} (+/- {results['cv_std']*2:.3f})")
    print(f"Test set R² score: {results['test_r2']:.3f}")
    print(f"Test set RMSE: {results['test_rmse']:.3f}")
    
    # Plot feature importance
    predictor.plot_feature_importance(results['feature_importance'])
    plt.show()

if __name__ == "__main__":
    main()

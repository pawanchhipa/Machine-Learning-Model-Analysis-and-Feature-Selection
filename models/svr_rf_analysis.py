"""
Support Vector Regression and Random Forest analysis implementation
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

def optimize_svr(X, y, param_grid=None):
    """
    Optimize SVR hyperparameters using GridSearchCV
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 0.25, 0.5, 1, 10],
            'epsilon': [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 5],
            'kernel': ['linear'],
            'gamma': ['scale', 'auto']
        }
    
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, 
                             scoring='neg_mean_squared_error', 
                             n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    return SVR(**grid_search.best_params_)

def optimize_rf(X, y, param_grid=None):
    """
    Optimize Random Forest hyperparameters using GridSearchCV
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    param_grid : dict, optional
        Grid of parameters to search
        
    Returns:
    --------
    RandomForestRegressor
        Optimized random forest model
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [5, 10, 50],
            'max_features': [1],
            'max_depth': [2, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                             cv=5, scoring='neg_mean_squared_error',
                             n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    return RandomForestRegressor(**grid_search.best_params_)

def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained random forest model
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained random forest model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importances
    """
    importances = pd.DataFrame(
        {'feature': feature_names,
         'importance': model.feature_importances_}
    )
    return importances.sort_values('importance', ascending=False)
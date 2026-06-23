import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress warnings from models that fail to converge
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# No 'optuna' or 'functools' import at the top level

def tune_model(X_train, y_train, model_class:str, search_space: list, static_settings:dict, n_trials=30):
    """
    Runs generic hyperparameter tuning for any sklearn-compatible model.
    
    If 'optuna' is not installed, this function will raise a warning,
    skip tuning, and return the 'static_settings' dictionary.
    
    Args:
        X_train (array): Training features.
        y_train (array): Training target.
        model_class (class): The model class to be tuned (e.g., MLPRegressor).
        search_space (list): A list of dicts defining the search space.
        static_settings (dict): A dict of fixed hyperparameters.
        n_trials (int): Number of Optuna trials to run.
        
    Returns:
        dict: The best set of *all* hyperparameters (static + tuned).
    """
    
    try:
        # --- 1. Import dependencies *inside* the function ---
        import optuna
        import functools
        
        # --- 2. Define the objective function *inside* tune_model ---
        # This keeps it out of the global scope and ensures it's only
        # defined when optuna is actually available.
        def _objective(trial, X, y, model_class, search_space, static_settings):
            """
            Internal objective function for Optuna.
            This function is called by 'tune_model'.
            """
            
            # 1. Build the dictionary of tuned parameters
            tuned_params = {}
            for param_name, config in search_space.items():
                # Get param name and config
                # param_name = list(item.keys())[0]
                # config = search_space[param_name]
                
                if 'choices' in config:
                    # --- Categorical parameter ---
                    value = trial.suggest_categorical(param_name, config['choices'])
                
                elif 'range' in config:
                    # --- Numerical parameter (int or float) ---
                    low, high = config['range']
                    log = config.get('log', False)
                    
                    if isinstance(low, int) and isinstance(high, int):
                        # Integer range
                        value = trial.suggest_int(param_name, low, high, log=log)
                    else:
                        # Float range
                        value = trial.suggest_float(param_name, float(low), float(high), log=log)
                else:
                    raise ValueError(f"Invalid config in search_space for '{param_name}': {config}")
                    
                tuned_params[param_name] = value

            # 2. Create the final set of all hyperparameters
            model_params = static_settings.copy()
            model_params.update(tuned_params)
            
            # 3. Instantiate model and evaluate
            try:
                # Create a new model instance for this trial
                model_instance = model_class(**model_params)
                
                # Evaluate using 3-fold cross-validation
                score = cross_val_score(
                    model_instance, 
                    X, 
                    y, 
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                ).mean()
                
                # Optuna minimizes, so we return the positive MSE
                return -score 
            
            except Exception as e:
                # Handle cases where a hyperparameter combination is invalid
                print(f"Trial pruned. Error: {e}. Params: {model_params}")
                return float('inf')

        # --- 3. Run the tuning process ---
        print(f"--- Starting Optuna tuning for {model_class.__class__} ({n_trials} trials) ---")
        
        # Use functools.partial to pass the extra arguments to the objective
        objective_with_data = functools.partial(
            _objective, 
            X=X_train, 
            y=y_train, 
            model_class=model_class,
            search_space=search_space,
            static_settings=static_settings
        )

        # Create a study and minimize the objective (MSE)
        study = optuna.create_study(direction='minimize')
        
        # Suppress trial-by-trial logging
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        
        study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)
        
        # --- 4. Process Results ---
        print("\nTuning finished.")
        print(f"Best trial's score (MSE): {study.best_value}")
        
        # Combine static settings with the best-found params
        best_hyperparams = static_settings.copy()
        best_hyperparams.update(study.best_params)
        
        print("Best hyperparameters found:")
        for key, value in best_hyperparams.items():
            print(f"  - {key}: {value}")
            
        return best_hyperparams

    except ImportError:
        # --- 5. Handle the case where Optuna is not found ---
        warnings.warn(
            "The 'optuna' library is not installed. "
            "Skipping hyperparameter tuning. "
            "Returning static_settings."
        )
        print(f"--- Skipping tuning for {model_class.__name__} (optuna not found) ---")
        return static_settings
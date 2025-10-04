"""
Safe Model Training Script - Workaround for Python 3.13 numpy issues
"""

import warnings
warnings.filterwarnings('ignore')

# Set environment variables before importing numpy
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

try:
    # Import with error handling
    import sys
    import json
    from pathlib import Path
    
    print("Loading required libraries...")
    import pandas as pd
    import joblib
    from datetime import datetime
    
    # Try importing sklearn components individually
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Import XGBoost and LightGBM with fallback
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
    except:
        print("Warning: XGBoost not available, using alternatives")
        HAS_XGBOOST = False
    
    try:
        import lightgbm as lgb
        HAS_LIGHTGBM = True
    except:
        print("Warning: LightGBM not available, using alternatives")
        HAS_LIGHTGBM = False

    # Configuration
    TARGETS = ['very_hot', 'very_cold', 'very_windy', 'very_wet', 'very_uncomfortable']
    RANDOM_STATE = 42
    
    # Load data
    print("\nLoading engineered features...")
    data_path = Path('data/processed/features_engineered.csv')
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        print("Please run feature engineering first.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in TARGETS + ['latitude', 'longitude', 'date']]
    print(f"✓ Using {len(feature_cols)} features for training")
    
    # Create models directory
    models_dir = Path('models/trained')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature names
    joblib.dump(feature_cols, models_dir / 'feature_names.pkl')
    
    # Train models for each target
    results = {}
    
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"Training models for: {target}")
        print('='*60)
        
        # Prepare data
        X = df[feature_cols]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'logistic': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE
            )
        }
        
        # Add XGBoost and LightGBM if available
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric='logloss'
            )
        
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100, random_state=RANDOM_STATE, verbose=-1
            )
        
        # Train and evaluate models
        best_score = 0
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use scaled data for logistic regression
                if name == 'logistic':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # Calculate AUC if possible
                try:
                    if name == 'logistic':
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except:
                    auc = None
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                if auc:
                    print(f"  AUC-ROC: {auc:.4f}")
                
                # Track best model
                score = f1  # Use F1 score as main metric
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                continue
        
        # Save best model
        if best_model:
            model_filename = f"{target}_{best_model_name}.pkl"
            if best_model_name == 'logistic':
                # Save scaler with logistic regression
                joblib.dump((best_model, scaler), models_dir / model_filename)
            else:
                joblib.dump(best_model, models_dir / model_filename)
            
            print(f"\n✓ Best model for {target}: {best_model_name} (F1: {best_score:.4f})")
            print(f"✓ Saved to: models/trained/{model_filename}")
            
            results[target] = {
                'best_model': best_model_name,
                'f1_score': best_score
            }
    
    # Save metadata
    metadata = {
        'targets': TARGETS,
        'feature_count': len(feature_cols),
        'training_date': datetime.now().isoformat(),
        'results': results,
        'python_version': sys.version,
        'models_available': {
            'xgboost': HAS_XGBOOST,
            'lightgbm': HAS_LIGHTGBM
        }
    }
    
    with open(models_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nSummary:")
    for target, info in results.items():
        print(f"  {target}: {info['best_model']} (F1: {info['f1_score']:.4f})")
    
    print(f"\nAll models saved to: {models_dir}")
    print("\nNext steps:")
    print("  1. Run evaluation: python src/evaluate.py")
    print("  2. Start API: python src/api.py")
    print("  3. Open frontend: frontend/index.html")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nThis might be due to Python 3.13 compatibility issues.")
    print("Consider using Python 3.11 or 3.10 for better compatibility.")
    sys.exit(1)

# Custom inference logic for real-time predictions
class FraudDetectionInference:
    def __init__(self, ensemble_model, scaler, feature_cols):
        self.model = ensemble_model
        self.scaler = scaler
        self.feature_cols = feature_cols
    
    def preprocess(self, transaction_data):
        # Extract relevant features
        features = pd.DataFrame([transaction_data])
        features = features[self.feature_cols]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        return pd.DataFrame(features_scaled, columns=self.feature_cols)
    
    def predict_transaction(self, transaction_data):
        # Preprocess the transaction
        processed_data = self.preprocess(transaction_data)
        
        # Get model prediction
        fraud_probability = self.model.predict_proba(processed_data)[0]
        
        # Add custom business logic
        result = {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(fraud_probability >= 0.5),
            'confidence': 'high' if abs(fraud_probability - 0.5) > 0.3 else 'low',
            'timestamp': pd.Timestamp.now()
        }
        
        return result

class ModelInterpretability:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        
    def compute_shap_values(self, X, sample_size=100):
        """Compute SHAP values for feature importance"""
        if isinstance(self.model, xgb.XGBClassifier):
            explainer = shap.TreeExplainer(self.model)
        else:
            # For neural network, use KernelExplainer
            sample_data = shap.sample(X, sample_size)
            explainer = shap.KernelExplainer(self.model.predict_proba, sample_data)
        
        self.shap_values = explainer.shap_values(X)
        return self.shap_values
    
    def get_feature_importance(self, X):
        """Get global feature importance"""
        importance_dict = {}
        
        if isinstance(self.model, (RandomForestClassifier, xgb.XGBClassifier)):
            # For tree-based models
            importances = self.model.feature_importances_
            for name, importance in zip(self.feature_names, importances):
                importance_dict[name] = float(importance)
                
        elif isinstance(self.model, FraudDetectionNN):
            # For neural network using integrated gradients
            ig = IntegratedGradients(self.model)
            attributions = ig.attribute(X, target=1)
            importances = torch.mean(torch.abs(attributions), dim=0)
            
            for name, importance in zip(self.feature_names, importances):
                importance_dict[name] = float(importance)
        
        return importance_dict
    
    def explain_prediction(self, instance, num_features=10):
        """Explain a single prediction"""
        if self.shap_values is None:
            self.compute_shap_values(instance.reshape(1, -1))
        
        explanation = {
            'feature_contributions': {},
            'top_features': []
        }
        
        # Get feature contributions
        for i, name in enumerate(self.feature_names):
            explanation['feature_contributions'][name] = float(self.shap_values[0][i])
        
        # Get top contributing features
        sorted_features = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        explanation['top_features'] = sorted_features[:num_features]
        
        return explanation

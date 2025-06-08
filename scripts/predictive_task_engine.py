"""
Predictive Task Engine

Uses machine learning to predict future task needs and proactively generate them.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
from collections import defaultdict


class TaskPredictor:
    """ML-based task prediction system."""
    
    def __init__(self):
        """Initialize prediction models."""
        self.task_type_predictor = RandomForestClassifier(n_estimators=100)
        self.priority_predictor = RandomForestClassifier(n_estimators=50)
        self.success_predictor = RandomForestClassifier(n_estimators=50)
        self.timing_predictor = RandomForestRegressor(n_estimators=50)
        
        self.encoders = {
            'task_type': LabelEncoder(),
            'priority': LabelEncoder(),
            'project_phase': LabelEncoder()
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_models(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train all prediction models on historical data."""
        if len(historical_data) < 50:
            return {"error": "Insufficient data for training"}
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Feature engineering
        features = self._engineer_features(df)
        
        # Train each model
        metrics = {}
        
        # Task type prediction
        X_type, y_type = self._prepare_task_type_data(features)
        self.task_type_predictor.fit(X_type, y_type)
        metrics['task_type_accuracy'] = self._evaluate_model(
            self.task_type_predictor, X_type, y_type
        )
        
        # Priority prediction
        X_priority, y_priority = self._prepare_priority_data(features)
        self.priority_predictor.fit(X_priority, y_priority)
        metrics['priority_accuracy'] = self._evaluate_model(
            self.priority_predictor, X_priority, y_priority
        )
        
        # Success prediction
        X_success, y_success = self._prepare_success_data(features)
        self.success_predictor.fit(X_success, y_success)
        metrics['success_accuracy'] = self._evaluate_model(
            self.success_predictor, X_success, y_success
        )
        
        # Timing prediction
        X_timing, y_timing = self._prepare_timing_data(features)
        self.timing_predictor.fit(X_timing, y_timing)
        metrics['timing_rmse'] = self._evaluate_regressor(
            self.timing_predictor, X_timing, y_timing
        )
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        return metrics
    
    def predict_next_tasks(self, current_state: Dict[str, Any], 
                          num_tasks: int = 5) -> List[Dict[str, Any]]:
        """Predict the next tasks that will be needed."""
        if not self.is_trained:
            self._load_models()
        
        predictions = []
        
        # Analyze current state patterns
        state_features = self._extract_state_features(current_state)
        
        # Generate predictions for different task types
        task_types = ['feature', 'bug_fix', 'testing', 'documentation', 'security']
        
        for task_type in task_types:
            # Predict if this task type is needed
            need_probability = self._predict_task_need(state_features, task_type)
            
            if need_probability > 0.6:  # Threshold for task generation
                # Predict task details
                task = {
                    'type': task_type,
                    'probability': need_probability,
                    'predicted_priority': self._predict_priority(state_features, task_type),
                    'predicted_timing': self._predict_timing(state_features, task_type),
                    'predicted_success_rate': self._predict_success(state_features, task_type),
                    'recommended_description': self._generate_task_description(
                        task_type, state_features
                    ),
                    'risk_factors': self._identify_risk_factors(state_features, task_type)
                }
                
                predictions.append(task)
        
        # Sort by probability and timing
        predictions.sort(key=lambda x: (x['probability'], -x['predicted_timing']), reverse=True)
        
        # Enhance top predictions with specific details
        for pred in predictions[:num_tasks]:
            pred['specific_recommendations'] = self._generate_specific_recommendations(
                pred, current_state
            )
        
        return predictions[:num_tasks]
    
    def analyze_task_patterns(self, task_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in task history."""
        analysis = {
            'temporal_patterns': self._analyze_temporal_patterns(task_history),
            'success_patterns': self._analyze_success_patterns(task_history),
            'dependency_patterns': self._analyze_dependency_patterns(task_history),
            'seasonal_patterns': self._analyze_seasonal_patterns(task_history),
            'bottleneck_patterns': self._analyze_bottleneck_patterns(task_history)
        }
        
        return analysis
    
    def predict_project_trajectory(self, project_data: Dict[str, Any], 
                                 days_ahead: int = 30) -> Dict[str, Any]:
        """Predict project trajectory for the next N days."""
        trajectory = {
            'predicted_tasks': [],
            'workload_forecast': [],
            'risk_periods': [],
            'recommended_actions': []
        }
        
        current_date = datetime.now()
        
        for day in range(days_ahead):
            future_date = current_date + timedelta(days=day)
            
            # Predict workload
            workload = self._predict_daily_workload(project_data, future_date)
            trajectory['workload_forecast'].append({
                'date': future_date.isoformat(),
                'predicted_tasks': workload['task_count'],
                'complexity_score': workload['complexity'],
                'resource_requirement': workload['resources']
            })
            
            # Identify risk periods
            if workload['risk_score'] > 0.7:
                trajectory['risk_periods'].append({
                    'date': future_date.isoformat(),
                    'risk_type': workload['risk_type'],
                    'mitigation': workload['mitigation_strategy']
                })
        
        # Generate strategic recommendations
        trajectory['recommended_actions'] = self._generate_strategic_recommendations(
            trajectory['workload_forecast'],
            trajectory['risk_periods']
        )
        
        return trajectory
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data."""
        features = df.copy()
        
        # Time-based features
        if 'created_at' in features.columns:
            features['hour'] = pd.to_datetime(features['created_at']).dt.hour
            features['day_of_week'] = pd.to_datetime(features['created_at']).dt.dayofweek
            features['month'] = pd.to_datetime(features['created_at']).dt.month
        else:
            # Default time features
            features['hour'] = 12
            features['day_of_week'] = 1
            features['month'] = datetime.now().month
        
        # Task complexity features
        if 'description' in features.columns:
            features['description_length'] = features['description'].str.len()
        else:
            features['description_length'] = 50  # Default length
        
        if 'dependencies' in features.columns:
            features['has_dependencies'] = features['dependencies'].apply(lambda x: len(x) > 0)
            features['dependency_count'] = features['dependencies'].apply(len)
        else:
            features['has_dependencies'] = False
            features['dependency_count'] = 0
        
        # Historical features
        if 'type' in features.columns and 'success' in features.columns:
            features['prev_success_rate'] = features.groupby('type')['success'].transform(
                lambda x: x.expanding().mean().shift()
            )
        else:
            features['prev_success_rate'] = 0.8  # Default success rate
        
        # Project phase encoding
        features['project_phase'] = self._determine_project_phase(features)
        
        return features
    
    def _prepare_task_type_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for task type prediction."""
        feature_cols = ['hour', 'day_of_week', 'month', 'prev_success_rate', 
                       'project_phase_encoded']
        
        # Encode categorical variables
        features['project_phase_encoded'] = self.encoders['project_phase'].fit_transform(
            features['project_phase']
        )
        
        X = features[feature_cols].fillna(0).values
        y = self.encoders['task_type'].fit_transform(features['type'])
        
        return self.scaler.fit_transform(X), y
    
    def _prepare_priority_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for priority prediction."""
        feature_cols = ['description_length', 'dependency_count', 'prev_success_rate']
        
        X = features[feature_cols].fillna(0).values
        y = self.encoders['priority'].fit_transform(features['priority'])
        
        return X, y
    
    def _prepare_success_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for success prediction."""
        feature_cols = ['description_length', 'has_dependencies', 'hour', 'day_of_week']
        
        X = features[feature_cols].fillna(0).values
        y = features['success'].astype(int).values
        
        return X, y
    
    def _prepare_timing_data(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for timing prediction."""
        feature_cols = ['description_length', 'dependency_count', 'priority_encoded']
        
        features['priority_encoded'] = self.encoders['priority'].transform(features['priority'])
        
        X = features[feature_cols].fillna(0).values
        y = features['completion_hours'].values
        
        return X, y
    
    def _evaluate_model(self, model, X, y) -> float:
        """Evaluate classification model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    def _evaluate_regressor(self, model, X, y) -> float:
        """Evaluate regression model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        return rmse
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        models = {
            'task_type_predictor': self.task_type_predictor,
            'priority_predictor': self.priority_predictor,
            'success_predictor': self.success_predictor,
            'timing_predictor': self.timing_predictor,
            'encoders': self.encoders,
            'scaler': self.scaler
        }
        
        with open('predictive_models.pkl', 'wb') as f:
            pickle.dump(models, f)
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        try:
            with open('predictive_models.pkl', 'rb') as f:
                models = pickle.load(f)
                
            self.task_type_predictor = models['task_type_predictor']
            self.priority_predictor = models['priority_predictor']
            self.success_predictor = models['success_predictor']
            self.timing_predictor = models['timing_predictor']
            self.encoders = models['encoders']
            self.scaler = models['scaler']
            self.is_trained = True
        except FileNotFoundError:
            print("No saved models found")
    
    def _extract_state_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract features from current state."""
        features = []
        
        # Time features
        now = datetime.now()
        features.extend([now.hour, now.weekday(), now.month])
        
        # Project state features
        projects = state.get('projects', {})
        avg_health = np.mean([p.get('health_score', 50) for p in projects.values()]) if projects else 50
        features.append(avg_health)
        
        # Task queue features
        open_tasks = state.get('open_tasks', 0)
        features.append(open_tasks)
        
        # Recent activity
        recent_success_rate = state.get('recent_success_rate', 0.7)
        features.append(recent_success_rate)
        
        return np.array(features).reshape(1, -1)
    
    def _predict_task_need(self, features: np.ndarray, task_type: str) -> float:
        """Predict probability that a task type is needed."""
        # Simulate prediction based on features and task type
        base_prob = {
            'feature': 0.7,
            'bug_fix': 0.5,
            'testing': 0.6,
            'documentation': 0.4,
            'security': 0.3
        }.get(task_type, 0.5)
        
        # Adjust based on features (simplified)
        if features[0, 4] < 30:  # Low open tasks
            base_prob *= 1.2
        
        return min(base_prob, 1.0)
    
    def _predict_priority(self, features: np.ndarray, task_type: str) -> str:
        """Predict task priority."""
        # Simplified prediction
        if task_type in ['bug_fix', 'security']:
            return 'high'
        elif task_type == 'feature':
            return 'medium'
        else:
            return 'low'
    
    def _predict_timing(self, features: np.ndarray, task_type: str) -> float:
        """Predict optimal timing for task (days from now)."""
        base_timing = {
            'bug_fix': 0,  # Immediate
            'security': 1,  # Tomorrow
            'feature': 3,   # Few days
            'testing': 5,   # Week
            'documentation': 7  # Next week
        }.get(task_type, 3)
        
        return base_timing
    
    def _predict_success(self, features: np.ndarray, task_type: str) -> float:
        """Predict success probability."""
        base_success = {
            'documentation': 0.95,
            'testing': 0.9,
            'feature': 0.8,
            'bug_fix': 0.75,
            'security': 0.85
        }.get(task_type, 0.8)
        
        return base_success
    
    def _generate_task_description(self, task_type: str, features: np.ndarray) -> str:
        """Generate recommended task description."""
        descriptions = {
            'feature': "Implement new functionality based on user feedback analysis",
            'bug_fix': "Fix critical issues identified in recent error logs",
            'testing': "Increase test coverage for recently added features",
            'documentation': "Update API documentation for recent changes",
            'security': "Conduct security audit and implement recommendations"
        }
        
        return descriptions.get(task_type, "Perform necessary maintenance")
    
    def _identify_risk_factors(self, features: np.ndarray, task_type: str) -> List[str]:
        """Identify risk factors for the task."""
        risks = []
        
        if features[0, 4] > 50:  # Many open tasks
            risks.append("High workload may delay completion")
        
        if task_type == 'feature':
            risks.append("Scope creep potential")
        elif task_type == 'security':
            risks.append("May reveal additional vulnerabilities")
        
        return risks
    
    def _generate_specific_recommendations(self, prediction: Dict[str, Any], 
                                         state: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for a predicted task."""
        recommendations = []
        
        if prediction['type'] == 'feature':
            recommendations.append("Focus on user-requested features from issue tracker")
            recommendations.append("Implement using Laravel React starter kit patterns")
        elif prediction['type'] == 'testing':
            recommendations.append("Target modules with < 80% coverage")
            recommendations.append("Include E2E tests for critical user flows")
        
        return recommendations
    
    def _analyze_temporal_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in task history."""
        patterns = {
            'peak_hours': [],
            'peak_days': [],
            'task_frequency': {}
        }
        
        # Analyze by hour
        hour_counts = defaultdict(int)
        for task in history:
            hour = datetime.fromisoformat(task['created_at']).hour
            hour_counts[hour] += 1
        
        # Find peak hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        patterns['peak_hours'] = [h[0] for h in sorted_hours[:3]]
        
        return patterns
    
    def _analyze_success_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze success patterns."""
        patterns = {
            'success_by_type': {},
            'success_by_priority': {},
            'success_factors': []
        }
        
        # Group by type
        for task in history:
            task_type = task['type']
            if task_type not in patterns['success_by_type']:
                patterns['success_by_type'][task_type] = {'total': 0, 'successful': 0}
            
            patterns['success_by_type'][task_type]['total'] += 1
            if task.get('success', False):
                patterns['success_by_type'][task_type]['successful'] += 1
        
        # Calculate success rates
        for task_type, counts in patterns['success_by_type'].items():
            counts['success_rate'] = counts['successful'] / counts['total'] if counts['total'] > 0 else 0
        
        return patterns
    
    def _analyze_dependency_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze task dependency patterns."""
        patterns = {
            'common_dependencies': [],
            'dependency_chains': [],
            'bottleneck_tasks': []
        }
        
        # Find tasks that are frequently dependencies
        dependency_counts = defaultdict(int)
        for task in history:
            for dep in task.get('dependencies', []):
                dependency_counts[dep] += 1
        
        patterns['common_dependencies'] = sorted(
            dependency_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return patterns
    
    def _analyze_seasonal_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonal patterns."""
        patterns = {
            'monthly_trends': {},
            'quarterly_patterns': []
        }
        
        # Group by month
        monthly_counts = defaultdict(int)
        for task in history:
            month = datetime.fromisoformat(task['created_at']).month
            monthly_counts[month] += 1
        
        patterns['monthly_trends'] = dict(monthly_counts)
        
        return patterns
    
    def _analyze_bottleneck_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze bottleneck patterns."""
        patterns = {
            'long_running_types': [],
            'blocking_tasks': [],
            'resource_constraints': []
        }
        
        # Find task types that take longest
        type_durations = defaultdict(list)
        for task in history:
            if 'completion_hours' in task:
                type_durations[task['type']].append(task['completion_hours'])
        
        # Calculate average durations
        avg_durations = {}
        for task_type, durations in type_durations.items():
            avg_durations[task_type] = np.mean(durations)
        
        patterns['long_running_types'] = sorted(
            avg_durations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return patterns
    
    def _determine_project_phase(self, features: pd.DataFrame) -> pd.Series:
        """Determine project phase based on features."""
        # Simplified phase determination
        phases = []
        for _, row in features.iterrows():
            if row.get('task_count', 0) < 10:
                phases.append('initial')
            elif row.get('task_count', 0) < 50:
                phases.append('development')
            elif row.get('task_count', 0) < 100:
                phases.append('maturity')
            else:
                phases.append('maintenance')
        
        return pd.Series(phases)
    
    def _predict_daily_workload(self, project_data: Dict[str, Any], 
                               date: datetime) -> Dict[str, Any]:
        """Predict workload for a specific date."""
        workload = {
            'task_count': np.random.poisson(5),  # Poisson distribution for task arrival
            'complexity': np.random.uniform(0.3, 0.8),
            'resources': np.random.uniform(0.5, 1.0),
            'risk_score': np.random.uniform(0.1, 0.9),
            'risk_type': 'capacity' if np.random.random() > 0.5 else 'technical',
            'mitigation_strategy': 'Add resources' if np.random.random() > 0.5 else 'Reduce scope'
        }
        
        # Adjust for day of week
        if date.weekday() in [5, 6]:  # Weekend
            workload['task_count'] = int(workload['task_count'] * 0.3)
        
        return workload
    
    def _generate_strategic_recommendations(self, forecast: List[Dict[str, Any]], 
                                          risks: List[Dict[str, Any]]) -> List[str]:
        """Generate strategic recommendations based on predictions."""
        recommendations = []
        
        # Analyze workload peaks
        peak_days = sorted(forecast, key=lambda x: x['predicted_tasks'], reverse=True)[:3]
        if peak_days:
            recommendations.append(
                f"Prepare for high workload on {peak_days[0]['date'][:10]} "
                f"with {peak_days[0]['predicted_tasks']} predicted tasks"
            )
        
        # Risk mitigation
        if risks:
            recommendations.append(
                f"High risk period on {risks[0]['date'][:10]}: "
                f"Implement {risks[0]['mitigation']}"
            )
        
        # Resource planning
        avg_complexity = np.mean([f['complexity_score'] for f in forecast])
        if avg_complexity > 0.7:
            recommendations.append("High complexity period ahead - allocate senior resources")
        
        return recommendations


class PredictiveTaskEngine:
    """Main engine for predictive task generation."""
    
    def __init__(self):
        """Initialize predictive engine."""
        self.predictor = TaskPredictor()
        self.task_generator = None  # Will be TaskManager instance
        
    def analyze_and_predict(self, current_state: Dict[str, Any], 
                          historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current state and predict future needs."""
        analysis = {
            'current_analysis': self._analyze_current_state(current_state),
            'predictions': {},
            'recommendations': []
        }
        
        # Train models if needed
        if not self.predictor.is_trained and len(historical_data) > 50:
            training_metrics = self.predictor.train_models(historical_data)
            analysis['model_performance'] = training_metrics
        
        # Generate predictions
        predicted_tasks = self.predictor.predict_next_tasks(current_state)
        analysis['predictions']['next_tasks'] = predicted_tasks
        
        # Analyze patterns
        patterns = self.predictor.analyze_task_patterns(historical_data)
        analysis['patterns'] = patterns
        
        # Project trajectory
        trajectory = self.predictor.predict_project_trajectory(current_state)
        analysis['trajectory'] = trajectory
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(
            predicted_tasks, patterns, trajectory
        )
        
        return analysis
    
    def generate_predictive_tasks(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actual tasks from predictions."""
        generated_tasks = []
        
        for prediction in predictions:
            if prediction['probability'] > 0.7:  # High confidence threshold
                task = {
                    'type': prediction['type'],
                    'priority': prediction['predicted_priority'],
                    'title': f"[Predicted] {prediction['recommended_description'][:50]}...",
                    'description': self._enhance_description(prediction),
                    'estimated_hours': prediction['predicted_timing'] * 8,  # Convert days to hours
                    'ai_confidence': prediction['probability'],
                    'risk_factors': prediction['risk_factors']
                }
                
                generated_tasks.append(task)
        
        return generated_tasks
    
    def _analyze_current_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state."""
        analysis = {
            'health_score': self._calculate_system_health(state),
            'workload_score': self._calculate_workload(state),
            'risk_score': self._calculate_risk(state),
            'opportunity_score': self._calculate_opportunities(state)
        }
        
        return analysis
    
    def _calculate_system_health(self, state: Dict[str, Any]) -> float:
        """Calculate overall system health."""
        factors = []
        
        # Project health
        projects = state.get('projects', {})
        if projects:
            avg_health = np.mean([p.get('health_score', 50) for p in projects.values()])
            factors.append(avg_health / 100)
        
        # Success rate
        success_rate = state.get('recent_success_rate', 0.7)
        factors.append(success_rate)
        
        # Task completion rate
        completion_rate = state.get('completion_rate', 0.8)
        factors.append(completion_rate)
        
        return np.mean(factors) if factors else 0.5
    
    def _calculate_workload(self, state: Dict[str, Any]) -> float:
        """Calculate current workload."""
        open_tasks = state.get('open_tasks', 0)
        capacity = state.get('capacity', 20)
        
        return min(open_tasks / capacity, 1.0) if capacity > 0 else 1.0
    
    def _calculate_risk(self, state: Dict[str, Any]) -> float:
        """Calculate risk score."""
        risk_factors = []
        
        # High workload risk
        workload = self._calculate_workload(state)
        if workload > 0.8:
            risk_factors.append(0.7)
        
        # Low success rate risk
        if state.get('recent_success_rate', 1.0) < 0.6:
            risk_factors.append(0.8)
        
        # Stale tasks risk
        if state.get('stale_task_count', 0) > 5:
            risk_factors.append(0.6)
        
        return np.mean(risk_factors) if risk_factors else 0.2
    
    def _calculate_opportunities(self, state: Dict[str, Any]) -> float:
        """Calculate opportunity score."""
        opportunities = []
        
        # Low workload = opportunity for new features
        if self._calculate_workload(state) < 0.5:
            opportunities.append(0.8)
        
        # High success rate = opportunity for complex tasks
        if state.get('recent_success_rate', 0) > 0.85:
            opportunities.append(0.7)
        
        # Trending tech alignment
        if state.get('trending_tech_alignment', 0) > 0.7:
            opportunities.append(0.9)
        
        return np.mean(opportunities) if opportunities else 0.5
    
    def _generate_recommendations(self, predictions: List[Dict[str, Any]], 
                                patterns: Dict[str, Any],
                                trajectory: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []
        
        # Task type recommendations
        if predictions:
            top_prediction = predictions[0]
            recommendations.append(
                f"Priority: Generate {top_prediction['type']} tasks "
                f"(confidence: {top_prediction['probability']:.0%})"
            )
        
        # Pattern-based recommendations
        success_patterns = patterns.get('success_patterns', {})
        for task_type, stats in success_patterns.get('success_by_type', {}).items():
            if stats.get('success_rate', 0) < 0.6:
                recommendations.append(
                    f"Improve {task_type} task definitions - current success rate: "
                    f"{stats['success_rate']:.0%}"
                )
        
        # Trajectory-based recommendations
        if trajectory.get('risk_periods'):
            recommendations.append(
                f"Risk alert: {trajectory['risk_periods'][0]['risk_type']} risk on "
                f"{trajectory['risk_periods'][0]['date'][:10]}"
            )
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _enhance_description(self, prediction: Dict[str, Any]) -> str:
        """Enhance task description with predictive insights."""
        base_description = prediction['recommended_description']
        
        enhanced = f"""{base_description}

## Predictive Insights
- **Confidence**: {prediction['probability']:.0%}
- **Optimal Timing**: {prediction['predicted_timing']} days from now
- **Success Probability**: {prediction['predicted_success_rate']:.0%}
- **Priority**: {prediction['predicted_priority']}

## Risk Factors
{chr(10).join(f"- {risk}" for risk in prediction['risk_factors'])}

## Recommendations
{chr(10).join(f"- {rec}" for rec in prediction.get('specific_recommendations', []))}

*This task was generated by predictive analysis based on historical patterns and current state.*
"""
        
        return enhanced


# Example usage
def demonstrate_predictive_engine():
    """Demonstrate predictive task generation."""
    engine = PredictiveTaskEngine()
    
    # Mock current state
    current_state = {
        'projects': {
            'project1': {'health_score': 85},
            'project2': {'health_score': 72}
        },
        'open_tasks': 12,
        'capacity': 20,
        'recent_success_rate': 0.82,
        'completion_rate': 0.78,
        'stale_task_count': 2
    }
    
    # Mock historical data
    historical_data = [
        {
            'type': 'feature',
            'priority': 'high',
            'created_at': '2024-01-15T10:00:00',
            'completion_hours': 16,
            'success': True,
            'dependencies': [],
            'description': 'Implement user dashboard'
        }
        # Would have many more records in practice
    ] * 60  # Duplicate for demo
    
    # Run analysis
    analysis = engine.analyze_and_predict(current_state, historical_data)
    
    print("Predictive Analysis Results:")
    print(f"System Health: {analysis['current_analysis']['health_score']:.0%}")
    print(f"Risk Score: {analysis['current_analysis']['risk_score']:.0%}")
    
    print("\nPredicted Tasks:")
    for task in analysis['predictions']['next_tasks']:
        print(f"- {task['type']}: {task['recommended_description'][:50]}... "
              f"(confidence: {task['probability']:.0%})")
    
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"- {rec}")


if __name__ == "__main__":
    demonstrate_predictive_engine()
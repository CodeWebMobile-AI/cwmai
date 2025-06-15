"""
Task Intelligence Dashboard - Visualization and Monitoring

This module provides a comprehensive dashboard for visualizing task generation patterns,
prediction accuracy, and system learning progress.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

from intelligent_task_generator import IntelligentTaskGenerator
from progressive_task_generator import ProgressiveTaskGenerator
from predictive_task_engine import PredictiveTaskEngine
from smart_context_aggregator import SmartContextAggregator


@dataclass
class DashboardMetrics:
    """Metrics for dashboard display."""
    task_generation_rate: float  # Tasks per hour
    prediction_accuracy: float  # 0-1
    pattern_diversity: float  # 0-1, how diverse are patterns
    learning_progress: float  # 0-1, how much system has learned
    system_health: float  # 0-1, overall system health
    context_quality: float  # 0-1, quality of context data
    cross_repo_efficiency: float  # 0-1, cross-repo pattern usage
    adaptation_rate: float  # 0-1, how fast system adapts


@dataclass
class TrendData:
    """Time series data for trends."""
    timestamps: List[datetime]
    values: List[float]
    metric_name: str
    trend_direction: str  # 'up', 'down', 'stable'
    forecast: Optional[List[float]] = None


class TaskIntelligenceDashboard:
    """Dashboard for visualizing task intelligence system."""
    
    def __init__(self, 
                 task_generator: IntelligentTaskGenerator,
                 progressive_generator: Optional[ProgressiveTaskGenerator] = None,
                 predictive_engine: Optional[PredictiveTaskEngine] = None,
                 context_aggregator: Optional[SmartContextAggregator] = None,
                 output_dir: str = "dashboard_output"):
        """Initialize dashboard.
        
        Args:
            task_generator: Intelligent task generator
            progressive_generator: Progressive task generator
            predictive_engine: Predictive engine
            context_aggregator: Context aggregator
            output_dir: Directory for dashboard outputs
        """
        self.task_generator = task_generator
        self.progressive_generator = progressive_generator
        self.predictive_engine = predictive_engine
        self.context_aggregator = context_aggregator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.prediction_history = []
        self.pattern_evolution = defaultdict(list)
        
        # Set style for plots
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    async def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard with all visualizations.
        
        Returns:
            Dashboard data and file paths
        """
        self.logger.info("Generating task intelligence dashboard")
        
        # Collect current metrics
        metrics = await self._collect_current_metrics()
        
        # Generate visualizations
        viz_paths = {
            'task_generation_patterns': await self._visualize_task_patterns(),
            'prediction_accuracy': await self._visualize_prediction_accuracy(),
            'learning_progress': await self._visualize_learning_progress(),
            'system_health': await self._visualize_system_health(),
            'cross_repo_insights': await self._visualize_cross_repo_insights(),
            'trend_analysis': await self._visualize_trends()
        }
        
        # Generate summary report
        summary = await self._generate_summary_report(metrics)
        
        # Save dashboard data
        dashboard_data = {
            'metrics': metrics.__dict__,
            'visualizations': viz_paths,
            'summary': summary,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        dashboard_path = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        self.logger.info(f"Dashboard generated at {dashboard_path}")
        return dashboard_data
    
    async def _collect_current_metrics(self) -> DashboardMetrics:
        """Collect current system metrics.
        
        Returns:
            Current dashboard metrics
        """
        # Get generation analytics
        gen_analytics = self.task_generator.get_generation_analytics()
        
        # Calculate task generation rate
        recent_tasks = gen_analytics.get('recent_tasks', [])
        if recent_tasks:
            # Assuming tasks have timestamps
            task_rate = len(recent_tasks) / 24.0  # Tasks per hour over last day
        else:
            task_rate = 0.0
        
        # Get prediction accuracy
        pred_accuracy = 0.0
        if self.predictive_engine:
            confidence = self.predictive_engine.get_prediction_confidence()
            pred_accuracy = confidence.get('overall', 0.0)
        
        # Calculate pattern diversity
        patterns = gen_analytics.get('generation_patterns', {})
        pattern_diversity = len(patterns) / 10.0  # Normalize to 0-1
        pattern_diversity = min(1.0, pattern_diversity)
        
        # Get learning progress
        learning_progress = 0.5  # Default
        if self.progressive_generator:
            prog_analytics = self.progressive_generator.get_progression_analytics()
            if prog_analytics.get('overall_success_rate'):
                learning_progress = prog_analytics['overall_success_rate']
        
        # Get context quality
        context_quality = 0.7  # Default
        if self.context_aggregator:
            quality_report = await self.context_aggregator.get_context_quality_report()
            context_quality = quality_report.get('overall_quality', 0.7)
        
        # Calculate system health
        system_health = np.mean([pred_accuracy, pattern_diversity, learning_progress, context_quality])
        
        # Cross-repo efficiency
        cross_repo_efficiency = 0.0
        if self.progressive_generator:
            prog_analytics = self.progressive_generator.get_progression_analytics()
            cross_insights = prog_analytics.get('cross_project_insights', {})
            if cross_insights:
                cross_repo_efficiency = cross_insights.get('success_rate', 0.0)
        
        # Adaptation rate (how fast patterns are updated)
        adaptation_rate = 0.6  # Placeholder - would calculate from pattern update frequency
        
        metrics = DashboardMetrics(
            task_generation_rate=task_rate,
            prediction_accuracy=pred_accuracy,
            pattern_diversity=pattern_diversity,
            learning_progress=learning_progress,
            system_health=system_health,
            context_quality=context_quality,
            cross_repo_efficiency=cross_repo_efficiency,
            adaptation_rate=adaptation_rate
        )
        
        # Store in history
        self.metrics_history['timestamp'].append(datetime.now(timezone.utc))
        for field, value in metrics.__dict__.items():
            self.metrics_history[field].append(value)
        
        return metrics
    
    async def _visualize_task_patterns(self) -> str:
        """Visualize task generation patterns.
        
        Returns:
            Path to visualization
        """
        analytics = self.task_generator.get_generation_analytics()
        patterns = analytics.get('generation_patterns', {})
        
        if not patterns:
            return self._create_empty_plot("No task patterns available")
        
        # Create pattern visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pattern distribution
        pattern_types = list(patterns.keys())
        pattern_counts = [len(patterns[p]) for p in pattern_types]
        
        ax1.bar(pattern_types, pattern_counts)
        ax1.set_xlabel('Pattern Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Task Generation Pattern Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pattern timeline
        all_patterns = []
        for pattern_type, entries in patterns.items():
            for entry in entries:
                if isinstance(entry, dict) and 'timestamp' in entry:
                    all_patterns.append({
                        'type': pattern_type,
                        'timestamp': entry['timestamp']
                    })
        
        if all_patterns:
            df = pd.DataFrame(all_patterns)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by hour and type
            hourly = df.groupby([pd.Grouper(key='timestamp', freq='H'), 'type']).size().unstack(fill_value=0)
            
            hourly.plot(ax=ax2, kind='area', stacked=True)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Pattern Count')
            ax2.set_title('Pattern Generation Over Time')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'task_patterns.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _visualize_prediction_accuracy(self) -> str:
        """Visualize prediction accuracy over time.
        
        Returns:
            Path to visualization
        """
        if not self.predictive_engine:
            return self._create_empty_plot("No predictive engine available")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Model performance metrics
        performance = self.predictive_engine.model_performance
        if performance:
            metrics = ['task_type_accuracy', 'urgency_accuracy', 'success_accuracy']
            values = [performance.get(m, 0) for m in metrics]
            
            ax1.bar(metrics, values)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Performance Metrics')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax1.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        # Prediction confidence over time
        if 'prediction_accuracy' in self.metrics_history:
            timestamps = self.metrics_history['timestamp']
            accuracies = self.metrics_history['prediction_accuracy']
            
            if len(timestamps) > 1:
                ax2.plot(timestamps, accuracies, marker='o')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Prediction Accuracy')
                ax2.set_title('Prediction Accuracy Trend')
                ax2.set_ylim(0, 1)
                
                # Add trend line
                z = np.polyfit(range(len(accuracies)), accuracies, 1)
                p = np.poly1d(z)
                ax2.plot(timestamps, p(range(len(accuracies))), "r--", alpha=0.5, label='Trend')
                ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'prediction_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _visualize_learning_progress(self) -> str:
        """Visualize system learning progress.
        
        Returns:
            Path to visualization
        """
        if not self.progressive_generator:
            return self._create_empty_plot("No progressive generator available")
        
        analytics = self.progressive_generator.get_progression_analytics()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top patterns
        top_patterns = analytics.get('top_patterns', [])
        if top_patterns:
            pattern_names = [p['name'][:30] + '...' if len(p['name']) > 30 else p['name'] 
                           for p in top_patterns]
            success_rates = [p['success_rate'] for p in top_patterns]
            
            ax1.barh(pattern_names, success_rates)
            ax1.set_xlabel('Success Rate')
            ax1.set_title('Top Progression Patterns')
            ax1.set_xlim(0, 1)
        
        # Learning metrics
        learning_metrics = analytics.get('learning_metrics', {})
        if learning_metrics:
            metrics = list(learning_metrics.keys())
            values = [1 if v else 0 for v in learning_metrics.values()]
            
            ax2.bar(metrics, values)
            ax2.set_ylim(0, 1.2)
            ax2.set_ylabel('Enabled')
            ax2.set_title('Learning Features Status')
            ax2.tick_params(axis='x', rotation=45)
        
        # Pattern evolution
        if 'learning_progress' in self.metrics_history and len(self.metrics_history['learning_progress']) > 1:
            timestamps = self.metrics_history['timestamp']
            progress = self.metrics_history['learning_progress']
            
            ax3.plot(timestamps, progress, marker='o', linewidth=2)
            ax3.fill_between(timestamps, progress, alpha=0.3)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Learning Progress')
            ax3.set_title('Learning Progress Over Time')
            ax3.set_ylim(0, 1)
        
        # Pattern count growth
        total_patterns = analytics.get('total_patterns', 0)
        ax4.text(0.5, 0.5, f'Total Patterns\n{total_patterns}', 
                fontsize=24, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Pattern Library Size')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'learning_progress.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _visualize_system_health(self) -> str:
        """Visualize overall system health.
        
        Returns:
            Path to visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Current health radar chart
        metrics = await self._collect_current_metrics()
        
        categories = ['Task Rate', 'Prediction', 'Diversity', 'Learning', 
                     'Context', 'Cross-Repo', 'Adaptation']
        values = [
            min(1.0, metrics.task_generation_rate / 10),  # Normalize to 0-1
            metrics.prediction_accuracy,
            metrics.pattern_diversity,
            metrics.learning_progress,
            metrics.context_quality,
            metrics.cross_repo_efficiency,
            metrics.adaptation_rate
        ]
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax1 = plt.subplot(121, projection='polar')
        ax1.plot(angles, values, 'o-', linewidth=2)
        ax1.fill(angles, values, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('System Health Radar', y=1.08)
        
        # Health trend
        if 'system_health' in self.metrics_history and len(self.metrics_history['system_health']) > 1:
            timestamps = self.metrics_history['timestamp']
            health_values = self.metrics_history['system_health']
            
            ax2.plot(timestamps, health_values, marker='o', linewidth=2, color='green')
            ax2.fill_between(timestamps, health_values, alpha=0.3, color='green')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('System Health')
            ax2.set_title('System Health Trend')
            ax2.set_ylim(0, 1)
            
            # Add health zones
            ax2.axhspan(0.8, 1.0, alpha=0.1, color='green', label='Excellent')
            ax2.axhspan(0.6, 0.8, alpha=0.1, color='yellow', label='Good')
            ax2.axhspan(0.4, 0.6, alpha=0.1, color='orange', label='Fair')
            ax2.axhspan(0.0, 0.4, alpha=0.1, color='red', label='Poor')
            ax2.legend(loc='best')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'system_health.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _visualize_cross_repo_insights(self) -> str:
        """Visualize cross-repository insights.
        
        Returns:
            Path to visualization
        """
        if not self.progressive_generator:
            return self._create_empty_plot("No cross-repo data available")
        
        analytics = self.progressive_generator.get_progression_analytics()
        cross_insights = analytics.get('cross_project_insights', {})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cross-repo metrics
        if cross_insights:
            metrics = ['Total Applications', 'Success Rate', 'Pattern Count']
            values = [
                cross_insights.get('total_cross_applications', 0),
                cross_insights.get('success_rate', 0) * 100,  # Convert to percentage
                cross_insights.get('pattern_count', 0)
            ]
            
            bars = ax1.bar(metrics, values)
            ax1.set_ylabel('Value')
            ax1.set_title('Cross-Repository Pattern Usage')
            
            # Color bars differently
            bars[0].set_color('blue')
            bars[1].set_color('green')
            bars[2].set_color('orange')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}' if isinstance(value, float) else str(value),
                        ha='center', va='bottom')
        
        # Cross-repo efficiency over time
        if 'cross_repo_efficiency' in self.metrics_history:
            timestamps = self.metrics_history['timestamp']
            efficiency = self.metrics_history['cross_repo_efficiency']
            
            ax2.plot(timestamps, efficiency, marker='o', linewidth=2, color='purple')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cross-Repo Efficiency')
            ax2.set_title('Cross-Repository Pattern Efficiency Trend')
            ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'cross_repo_insights.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _visualize_trends(self) -> str:
        """Visualize key metric trends.
        
        Returns:
            Path to visualization
        """
        if not self.metrics_history['timestamp']:
            return self._create_empty_plot("No historical data for trends")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Define metrics to plot
        metrics_to_plot = [
            ('task_generation_rate', 'Task Generation Rate', 'blue'),
            ('prediction_accuracy', 'Prediction Accuracy', 'green'),
            ('pattern_diversity', 'Pattern Diversity', 'orange'),
            ('adaptation_rate', 'Adaptation Rate', 'red')
        ]
        
        timestamps = self.metrics_history['timestamp']
        
        for idx, (metric, title, color) in enumerate(metrics_to_plot):
            if metric in self.metrics_history:
                values = self.metrics_history[metric]
                
                if len(values) > 1:
                    axes[idx].plot(timestamps, values, marker='o', color=color, linewidth=2)
                    
                    # Add trend line
                    z = np.polyfit(range(len(values)), values, 1)
                    p = np.poly1d(z)
                    axes[idx].plot(timestamps, p(range(len(values))), "--", color=color, alpha=0.5)
                    
                    # Determine trend direction
                    trend = "↑" if z[0] > 0.01 else "↓" if z[0] < -0.01 else "→"
                    
                    axes[idx].set_title(f'{title} {trend}')
                    axes[idx].set_xlabel('Time')
                    axes[idx].set_ylabel(title)
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Set y limits for percentage metrics
                    if metric in ['prediction_accuracy', 'pattern_diversity', 'adaptation_rate']:
                        axes[idx].set_ylim(0, 1)
        
        plt.suptitle('Key Metric Trends', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'metric_trends.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def _generate_summary_report(self, metrics: DashboardMetrics) -> Dict[str, Any]:
        """Generate summary report with insights and recommendations.
        
        Args:
            metrics: Current metrics
            
        Returns:
            Summary report
        """
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': self._determine_status(metrics.system_health),
            'key_metrics': {
                'system_health': f"{metrics.system_health:.1%}",
                'task_generation_rate': f"{metrics.task_generation_rate:.1f} tasks/hour",
                'prediction_accuracy': f"{metrics.prediction_accuracy:.1%}",
                'learning_progress': f"{metrics.learning_progress:.1%}"
            },
            'insights': [],
            'recommendations': [],
            'alerts': []
        }
        
        # Generate insights
        if metrics.prediction_accuracy > 0.8:
            report['insights'].append("Prediction models are performing excellently")
        
        if metrics.pattern_diversity > 0.7:
            report['insights'].append("High pattern diversity indicates good coverage")
        
        if metrics.cross_repo_efficiency > 0.6:
            report['insights'].append("Cross-repository patterns are being effectively utilized")
        
        # Generate recommendations
        if metrics.task_generation_rate < 1.0:
            report['recommendations'].append("Consider increasing task generation frequency")
        
        if metrics.context_quality < 0.6:
            report['recommendations'].append("Improve context data collection for better decisions")
        
        if metrics.adaptation_rate < 0.5:
            report['recommendations'].append("System adaptation is slow - review learning parameters")
        
        # Generate alerts
        if metrics.system_health < 0.4:
            report['alerts'].append({
                'severity': 'high',
                'message': 'System health is poor - immediate attention required'
            })
        
        if metrics.prediction_accuracy < 0.5:
            report['alerts'].append({
                'severity': 'medium',
                'message': 'Prediction accuracy is low - consider retraining models'
            })
        
        return report
    
    def _determine_status(self, health_score: float) -> str:
        """Determine overall status from health score.
        
        Args:
            health_score: System health score (0-1)
            
        Returns:
            Status string
        """
        if health_score >= 0.8:
            return "Excellent"
        elif health_score >= 0.6:
            return "Good"
        elif health_score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _create_empty_plot(self, message: str) -> str:
        """Create empty plot with message.
        
        Args:
            message: Message to display
            
        Returns:
            Path to plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plot_path = self.output_dir / f'empty_{datetime.now().timestamp()}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    async def start_real_time_monitoring(self, update_interval: int = 300):
        """Start real-time monitoring with periodic updates.
        
        Args:
            update_interval: Update interval in seconds (default: 5 minutes)
        """
        self.logger.info(f"Starting real-time monitoring with {update_interval}s interval")
        
        while True:
            try:
                # Generate dashboard
                await self.generate_dashboard()
                
                # Generate quick status
                metrics = await self._collect_current_metrics()
                status = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'health': metrics.system_health,
                    'status': self._determine_status(metrics.system_health),
                    'alerts': []
                }
                
                # Check for alerts
                if metrics.system_health < 0.5:
                    status['alerts'].append("System health below 50%")
                if metrics.prediction_accuracy < 0.6:
                    status['alerts'].append("Prediction accuracy degraded")
                
                # Save status
                status_path = self.output_dir / 'current_status.json'
                with open(status_path, 'w') as f:
                    json.dump(status, f, indent=2)
                
                self.logger.info(f"Dashboard updated. Health: {metrics.system_health:.1%}")
                
            except Exception as e:
                self.logger.error(f"Error in real-time monitoring: {e}")
            
            # Wait for next update
            await asyncio.sleep(update_interval)
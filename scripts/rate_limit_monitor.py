"""
Rate Limit Monitor Module

Real-time monitoring dashboard for API rate limiting with admin capabilities,
performance metrics, and alerting features.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import logging

try:
    from rate_limiter import RateLimiter, RateLimitTier
except ImportError:
    RateLimiter = None
    RateLimitTier = None


class RateLimitMonitor:
    """Real-time monitoring and admin dashboard for rate limiting."""
    
    def __init__(self):
        """Initialize the rate limit monitor."""
        self.logger = logging.getLogger(f"{__name__}.RateLimitMonitor")
        
        # Initialize rate limiter for monitoring
        self.rate_limiter = None
        if RateLimiter:
            try:
                self.rate_limiter = RateLimiter()
                self.logger.info("Rate limit monitor initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize rate limiter: {e}")
        else:
            self.logger.warning("RateLimiter module not available")
        
        # Monitor state
        self.start_time = datetime.now(timezone.utc)
        self.alert_thresholds = {
            "block_rate": 0.1,  # Alert if >10% requests blocked
            "error_rate": 0.05,  # Alert if >5% errors
            "high_usage_threshold": 0.8  # Alert if >80% of limit used
        }
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Generate real-time dashboard data.
        
        Returns:
            Comprehensive dashboard data
        """
        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monitor_uptime": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "system_status": "unknown",
            "alerts": [],
            "metrics": {},
            "top_clients": [],
            "recent_activity": [],
            "performance_stats": {},
            "configuration": {}
        }
        
        if not self.rate_limiter:
            dashboard["system_status"] = "unavailable"
            dashboard["alerts"].append({
                "level": "error",
                "message": "Rate limiter not available",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return dashboard
        
        try:
            # Get system metrics
            system_metrics = self.rate_limiter.get_system_metrics()
            dashboard["metrics"] = system_metrics
            
            # Determine system status
            if system_metrics.get("redis_available", False):
                if system_metrics.get("error_count", 0) == 0:
                    dashboard["system_status"] = "healthy"
                elif system_metrics.get("block_rate", 0) < self.alert_thresholds["block_rate"]:
                    dashboard["system_status"] = "warning"
                else:
                    dashboard["system_status"] = "critical"
            else:
                dashboard["system_status"] = "degraded"
                dashboard["alerts"].append({
                    "level": "warning",
                    "message": "Redis not available - using fallback mode",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Check for alerts
            dashboard["alerts"].extend(self._check_alerts(system_metrics))
            
            # Get performance statistics
            dashboard["performance_stats"] = self._calculate_performance_stats(system_metrics)
            
            # Get top clients by usage
            dashboard["top_clients"] = self._get_top_clients()
            
            # Get recent activity
            dashboard["recent_activity"] = system_metrics.get("recent_activity", [])[:10]
            
            # Get configuration info
            dashboard["configuration"] = self._get_configuration_info()
            
        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            dashboard["system_status"] = "error"
            dashboard["alerts"].append({
                "level": "error",
                "message": f"Dashboard error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return dashboard
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions.
        
        Args:
            metrics: System metrics
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check block rate
        block_rate = metrics.get("block_rate", 0)
        if block_rate > self.alert_thresholds["block_rate"]:
            alerts.append({
                "level": "warning" if block_rate < 0.2 else "critical",
                "message": f"High block rate: {block_rate:.1%}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metric": "block_rate",
                "value": block_rate
            })
        
        # Check error rate
        total_requests = metrics.get("total_requests", 0)
        error_count = metrics.get("error_count", 0)
        if total_requests > 0:
            error_rate = error_count / total_requests
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append({
                    "level": "warning" if error_rate < 0.1 else "critical",
                    "message": f"High error rate: {error_rate:.1%}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metric": "error_rate",
                    "value": error_rate
                })
        
        # Check for stale data (no recent requests)
        recent_activity = metrics.get("recent_activity", [])
        if recent_activity:
            last_activity = datetime.fromisoformat(recent_activity[0]["timestamp"].replace("Z", "+00:00"))
            minutes_since_activity = (datetime.now(timezone.utc) - last_activity).total_seconds() / 60
            
            if minutes_since_activity > 30:  # No activity for 30 minutes
                alerts.append({
                    "level": "info",
                    "message": f"No recent activity for {minutes_since_activity:.0f} minutes",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metric": "activity_age",
                    "value": minutes_since_activity
                })
        
        return alerts
    
    def _calculate_performance_stats(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance statistics.
        
        Args:
            metrics: System metrics
            
        Returns:
            Performance statistics
        """
        stats = {
            "requests_per_minute": 0,
            "average_response_time": 0,
            "success_rate": 0,
            "throughput": 0,
            "efficiency_score": 0
        }
        
        try:
            total_requests = metrics.get("total_requests", 0)
            uptime_seconds = metrics.get("uptime_seconds", 1)
            blocked_requests = metrics.get("blocked_requests", 0)
            allowed_requests = metrics.get("allowed_requests", 0)
            
            # Calculate requests per minute
            if uptime_seconds > 0:
                stats["requests_per_minute"] = (total_requests / uptime_seconds) * 60
            
            # Calculate success rate
            if total_requests > 0:
                stats["success_rate"] = allowed_requests / total_requests
            
            # Calculate throughput (allowed requests per minute)
            if uptime_seconds > 0:
                stats["throughput"] = (allowed_requests / uptime_seconds) * 60
            
            # Calculate efficiency score (combination of success rate and utilization)
            active_clients = metrics.get("active_clients", 0)
            if active_clients > 0 and total_requests > 0:
                utilization = min(1.0, total_requests / (active_clients * 100))  # Assume 100 req/client baseline
                stats["efficiency_score"] = (stats["success_rate"] * 0.7) + (utilization * 0.3)
            
            # Get response time from recent activity
            recent_activity = metrics.get("recent_activity", [])
            if recent_activity:
                response_times = [activity.get("duration_ms", 0) for activity in recent_activity]
                if response_times:
                    stats["average_response_time"] = sum(response_times) / len(response_times)
            
        except Exception as e:
            self.logger.error(f"Performance stats calculation failed: {e}")
        
        return stats
    
    def _get_top_clients(self) -> List[Dict[str, Any]]:
        """Get top clients by usage.
        
        Returns:
            List of top clients with their usage statistics
        """
        top_clients = []
        
        if not self.rate_limiter:
            return top_clients
        
        try:
            # This would require extending the rate limiter to track client usage
            # For now, return a placeholder implementation
            system_metrics = self.rate_limiter.get_system_metrics()
            recent_activity = system_metrics.get("recent_activity", [])
            
            # Aggregate by client
            client_usage = {}
            for activity in recent_activity:
                client_id = activity.get("client_id", "unknown")
                if client_id not in client_usage:
                    client_usage[client_id] = {
                        "client_id": client_id,
                        "total_requests": 0,
                        "blocked_requests": 0,
                        "tier": activity.get("tier", "unknown"),
                        "last_activity": activity.get("timestamp")
                    }
                
                client_usage[client_id]["total_requests"] += 1
                if not activity.get("allowed", True):
                    client_usage[client_id]["blocked_requests"] += 1
            
            # Sort by total requests and return top 10
            top_clients = sorted(
                client_usage.values(),
                key=lambda x: x["total_requests"],
                reverse=True
            )[:10]
            
            # Add block rate for each client
            for client in top_clients:
                if client["total_requests"] > 0:
                    client["block_rate"] = client["blocked_requests"] / client["total_requests"]
                else:
                    client["block_rate"] = 0
            
        except Exception as e:
            self.logger.error(f"Top clients calculation failed: {e}")
        
        return top_clients
    
    def _get_configuration_info(self) -> Dict[str, Any]:
        """Get rate limiting configuration information.
        
        Returns:
            Configuration details
        """
        config = {
            "redis_url": os.getenv('REDIS_URL', 'not_configured'),
            "rate_limit_enabled": os.getenv('RATE_LIMIT_ENABLED', 'true'),
            "default_tier": os.getenv('RATE_LIMIT_TIER', 'basic'),
            "tier_limits": {}
        }
        
        if self.rate_limiter and RateLimitTier:
            try:
                # Get default rules for each tier
                for tier in RateLimitTier:
                    rule = self.rate_limiter.rules.get(tier)
                    if rule:
                        config["tier_limits"][tier.value] = {
                            "requests_per_minute": rule.requests_per_minute,
                            "requests_per_hour": rule.requests_per_hour,
                            "requests_per_day": rule.requests_per_day,
                            "burst_allowance": rule.burst_allowance,
                            "strategy": rule.strategy.value
                        }
            except Exception as e:
                self.logger.error(f"Configuration info failed: {e}")
        
        return config
    
    def get_client_details(self, client_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Detailed client information
        """
        if not self.rate_limiter:
            return {"error": "Rate limiter not available"}
        
        try:
            return self.rate_limiter.get_client_stats(client_id)
        except Exception as e:
            self.logger.error(f"Client details failed for {client_id}: {e}")
            return {"error": str(e)}
    
    def update_client_tier(self, client_id: str, new_tier: str) -> Dict[str, Any]:
        """Update a client's rate limit tier.
        
        Args:
            client_id: Client identifier
            new_tier: New tier name
            
        Returns:
            Operation result
        """
        if not self.rate_limiter:
            return {"success": False, "error": "Rate limiter not available"}
        
        try:
            if not hasattr(RateLimitTier, new_tier.upper()):
                return {"success": False, "error": f"Invalid tier: {new_tier}"}
            
            tier_enum = getattr(RateLimitTier, new_tier.upper())
            success = self.rate_limiter.update_client_tier(client_id, tier_enum)
            
            return {
                "success": success,
                "client_id": client_id,
                "new_tier": new_tier,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Tier update failed for {client_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def reset_client_limits(self, client_id: str) -> Dict[str, Any]:
        """Reset rate limits for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Operation result
        """
        if not self.rate_limiter:
            return {"success": False, "error": "Rate limiter not available"}
        
        try:
            success = self.rate_limiter.reset_client_limits(client_id)
            
            return {
                "success": success,
                "client_id": client_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Reset failed for {client_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_usage_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a usage report for the specified time period.
        
        Args:
            hours: Number of hours to include in the report
            
        Returns:
            Usage report
        """
        report = {
            "report_period_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "trends": {},
            "recommendations": []
        }
        
        if not self.rate_limiter:
            report["error"] = "Rate limiter not available"
            return report
        
        try:
            # Get current metrics
            metrics = self.rate_limiter.get_system_metrics()
            
            # Calculate summary
            report["summary"] = {
                "total_requests": metrics.get("total_requests", 0),
                "allowed_requests": metrics.get("allowed_requests", 0),
                "blocked_requests": metrics.get("blocked_requests", 0),
                "block_rate": metrics.get("block_rate", 0),
                "active_clients": metrics.get("active_clients", 0),
                "uptime_hours": metrics.get("uptime_seconds", 0) / 3600
            }
            
            # Add trends (simplified - would need historical data for real trends)
            report["trends"] = {
                "requests_trend": "stable",  # Would calculate from historical data
                "block_rate_trend": "decreasing" if metrics.get("block_rate", 0) < 0.05 else "stable",
                "client_growth": "stable"
            }
            
            # Generate recommendations
            block_rate = metrics.get("block_rate", 0)
            if block_rate > 0.1:
                report["recommendations"].append(
                    "High block rate detected. Consider reviewing rate limit tiers for active clients."
                )
            
            if metrics.get("active_clients", 0) > 100:
                report["recommendations"].append(
                    "High number of active clients. Consider implementing adaptive rate limiting."
                )
            
            if not metrics.get("redis_available", False):
                report["recommendations"].append(
                    "Redis connection issues detected. Check Redis configuration and connectivity."
                )
            
            if len(report["recommendations"]) == 0:
                report["recommendations"].append("System is operating within normal parameters.")
            
        except Exception as e:
            self.logger.error(f"Usage report generation failed: {e}")
            report["error"] = str(e)
        
        return report
    
    def export_metrics(self, format: str = "json") -> str:
        """Export current metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            
        Returns:
            Exported metrics as string
        """
        try:
            dashboard_data = self.get_real_time_dashboard()
            
            if format.lower() == "json":
                return json.dumps(dashboard_data, indent=2)
            elif format.lower() == "csv":
                # Simplified CSV export
                lines = ["timestamp,metric,value"]
                timestamp = dashboard_data["timestamp"]
                
                metrics = dashboard_data.get("metrics", {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"{timestamp},{key},{value}")
                
                return "\n".join(lines)
            else:
                return "Unsupported format. Use 'json' or 'csv'."
                
        except Exception as e:
            self.logger.error(f"Metrics export failed: {e}")
            return f"Export failed: {str(e)}"


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rate Limit Monitor")
    parser.add_argument("command", choices=["dashboard", "client", "report", "export"],
                       help="Command to execute")
    parser.add_argument("--client-id", help="Client ID for client-specific commands")
    parser.add_argument("--tier", help="New tier for tier update")
    parser.add_argument("--hours", type=int, default=24, help="Hours for report")
    parser.add_argument("--format", default="json", help="Export format")
    
    args = parser.parse_args()
    
    monitor = RateLimitMonitor()
    
    if args.command == "dashboard":
        dashboard = monitor.get_real_time_dashboard()
        print(json.dumps(dashboard, indent=2))
    
    elif args.command == "client":
        if not args.client_id:
            print("Error: --client-id required for client command")
            return
        
        if args.tier:
            result = monitor.update_client_tier(args.client_id, args.tier)
            print(json.dumps(result, indent=2))
        else:
            details = monitor.get_client_details(args.client_id)
            print(json.dumps(details, indent=2))
    
    elif args.command == "report":
        report = monitor.generate_usage_report(args.hours)
        print(json.dumps(report, indent=2))
    
    elif args.command == "export":
        exported = monitor.export_metrics(args.format)
        print(exported)


if __name__ == "__main__":
    main()
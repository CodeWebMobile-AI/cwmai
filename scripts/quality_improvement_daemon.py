"""
Quality Improvement Daemon
==========================

Background process that continuously monitors and improves AI response quality.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from scripts.response_quality_monitor import ResponseQualityMonitor
from scripts.conversational_ai_logger import log_system_message


class QualityImprovementDaemon:
    """Background daemon for continuous quality improvement."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitor = ResponseQualityMonitor()
        self.running = False
        self.check_interval = 300  # 5 minutes
        
    async def start(self):
        """Start the quality improvement daemon."""
        self.running = True
        log_system_message("Quality Improvement Daemon started")
        
        while self.running:
            try:
                # Generate quality report
                report = await self.monitor.generate_quality_report()
                
                if report.get("status") != "no_data":
                    log_system_message(
                        f"Quality Report - Score: {report.get('average_quality_score', 'N/A')}, "
                        f"Trend: {report.get('trend', 'unknown')}"
                    )
                    
                    # Trigger improvements if needed
                    improvement_result = await self.monitor.trigger_system_improvement()
                    
                    if improvement_result.get("improvements_triggered"):
                        log_system_message(
                            f"Triggered quality improvements: {improvement_result.get('actions', [])}"
                        )
                        
                        # Update the assistant's behavior
                        await self._update_assistant_instructions(improvement_result.get('actions', []))
                        
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in quality daemon: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    async def stop(self):
        """Stop the daemon."""
        self.running = False
        log_system_message("Quality Improvement Daemon stopped")
        
    async def _update_assistant_instructions(self, improvements: list):
        """Update assistant instructions based on quality improvements."""
        # Create a dynamic instruction file that the assistant checks
        instructions = {
            "timestamp": datetime.now().isoformat(),
            "quality_reminders": improvements,
            "priority": "high",
            "apply_immediately": True
        }
        
        instruction_file = Path("current_quality_instructions.json")
        
        with open(instruction_file, 'w') as f:
            import json
            json.dump(instructions, f, indent=2)
            
        log_system_message(f"Updated assistant instructions with {len(improvements)} quality improvements")


# Singleton instance
_daemon_instance = None


def get_quality_daemon():
    """Get the singleton quality daemon instance."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = QualityImprovementDaemon()
    return _daemon_instance


async def start_quality_monitoring():
    """Start quality monitoring in the background."""
    daemon = get_quality_daemon()
    asyncio.create_task(daemon.start())
    return daemon
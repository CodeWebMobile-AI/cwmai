#!/usr/bin/env python3
"""
Review Staged Improvements CLI

Interactive command-line tool for reviewing and managing staged improvements.
"""

import os
import sys
import json
import argparse
import difflib
from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich.progress import track
import asyncio

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from staged_self_improver import StagedSelfImprover
from improvement_validator import ImprovementValidator
from staged_improvement_monitor import StagedImprovementMonitor


class ImprovementReviewer:
    """Interactive reviewer for staged improvements."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize reviewer with repository path."""
        self.repo_path = os.path.abspath(repo_path)
        self.console = Console()
        self.improver = StagedSelfImprover(repo_path)
        self.validator = ImprovementValidator(repo_path)
        self.monitor = StagedImprovementMonitor(repo_path)
    
    def run(self):
        """Run the interactive review interface."""
        self.console.print(Panel.fit(
            "[bold cyan]Staged Improvements Review Tool[/bold cyan]\n"
            "Review and manage staged code improvements",
            border_style="cyan"
        ))
        
        while True:
            try:
                self._show_main_menu()
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit review tool?[/yellow]"):
                    self.console.print("\n[green]Goodbye![/green]")
                    break
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
    
    def _show_main_menu(self):
        """Show main menu options."""
        self.console.print("\n[bold]Main Menu:[/bold]")
        self.console.print("1. List staged improvements")
        self.console.print("2. Review specific improvement")
        self.console.print("3. Validate improvements")
        self.console.print("4. Apply improvements")
        self.console.print("5. View monitoring reports")
        self.console.print("6. Generate summary report")
        self.console.print("7. Exit")
        
        choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6", "7"])
        
        if choice == "1":
            self._list_improvements()
        elif choice == "2":
            self._review_improvement()
        elif choice == "3":
            self._validate_improvements()
        elif choice == "4":
            self._apply_improvements()
        elif choice == "5":
            self._view_monitoring_reports()
        elif choice == "6":
            self._generate_summary_report()
        elif choice == "7":
            raise KeyboardInterrupt
    
    def _list_improvements(self):
        """List all staged improvements."""
        self.console.print("\n[bold]Staged Improvements:[/bold]")
        
        # Get improvements by status
        staged = self.improver.get_staged_improvements('staged')
        validated = self.improver.get_staged_improvements('validated')
        applied = self.improver.get_staged_improvements('applied')
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=20)
        table.add_column("File", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Created", style="white")
        
        all_improvements = [
            (imp, "staged") for imp in staged
        ] + [
            (imp, "validated") for imp in validated
        ] + [
            (imp, "applied") for imp in applied
        ]
        
        for imp, status in all_improvements:
            table.add_row(
                imp.metadata['staging_id'][:12] + "...",
                os.path.basename(imp.modification.target_file),
                imp.modification.type.value,
                status,
                imp.created_at.strftime("%Y-%m-%d %H:%M")
            )
        
        self.console.print(table)
        self.console.print(f"\nTotal: {len(all_improvements)} improvements")
    
    def _review_improvement(self):
        """Review a specific improvement."""
        staging_id = Prompt.ask("\nEnter staging ID (or 'back' to return)")
        
        if staging_id.lower() == 'back':
            return
        
        # Find improvement
        improvement = None
        for imp in self.improver.staged_improvements.values():
            if imp.metadata['staging_id'].startswith(staging_id):
                improvement = imp
                break
        
        if not improvement:
            self.console.print("[red]Improvement not found[/red]")
            return
        
        # Show improvement details
        self.console.print(f"\n[bold]Improvement Details:[/bold]")
        self.console.print(f"ID: [cyan]{improvement.metadata['staging_id']}[/cyan]")
        self.console.print(f"File: [green]{improvement.modification.target_file}[/green]")
        self.console.print(f"Type: [yellow]{improvement.modification.type.value}[/yellow]")
        self.console.print(f"Description: {improvement.modification.description}")
        self.console.print(f"Safety Score: [blue]{improvement.modification.safety_score:.2f}[/blue]")
        
        # Show diff
        if Confirm.ask("\nShow diff?"):
            self._show_diff(improvement)
        
        # Show validation status
        if improvement.validation_status:
            self._show_validation_status(improvement.validation_status)
        
        # Actions
        self.console.print("\n[bold]Actions:[/bold]")
        self.console.print("1. Validate this improvement")
        self.console.print("2. Apply this improvement")
        self.console.print("3. Reject this improvement")
        self.console.print("4. Back to menu")
        
        action = Prompt.ask("Select action", choices=["1", "2", "3", "4"])
        
        if action == "1":
            asyncio.run(self._validate_single(improvement.metadata['staging_id']))
        elif action == "2":
            asyncio.run(self._apply_single(improvement.metadata['staging_id']))
        elif action == "3":
            self._reject_improvement(improvement.metadata['staging_id'])
    
    def _show_diff(self, improvement):
        """Show diff for an improvement."""
        try:
            # Read original file
            with open(improvement.original_path, 'r') as f:
                original_lines = f.readlines()
            
            # Read staged file
            with open(improvement.staged_path, 'r') as f:
                staged_lines = f.readlines()
            
            # Generate diff
            diff = difflib.unified_diff(
                original_lines,
                staged_lines,
                fromfile=f"original/{os.path.basename(improvement.original_path)}",
                tofile=f"staged/{os.path.basename(improvement.staged_path)}",
                lineterm=''
            )
            
            # Display with syntax highlighting
            diff_text = '\n'.join(diff)
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            self.console.print("\n[bold]Diff:[/bold]")
            self.console.print(syntax)
            
        except Exception as e:
            self.console.print(f"[red]Error showing diff: {e}[/red]")
    
    def _show_validation_status(self, validation_status: Dict[str, Any]):
        """Show validation status details."""
        self.console.print("\n[bold]Validation Status:[/bold]")
        
        # Create status table
        table = Table(show_header=False)
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        
        checks = [
            ("Ready to Apply", "✅" if validation_status.get('ready_to_apply') else "❌"),
            ("Syntax Valid", "✅" if validation_status.get('syntax_valid') else "❌"),
            ("Tests Pass", "✅" if validation_status.get('tests_pass') else "❌"),
            ("Performance", "✅" if validation_status.get('performance_improved') else "⚠️"),
            ("Security", "✅" if validation_status.get('security_passed') else "❌"),
            ("Compatibility", "✅" if validation_status.get('compatibility_ok') else "⚠️")
        ]
        
        for check, status in checks:
            table.add_row(check, status)
        
        self.console.print(table)
        
        # Show errors/warnings
        if validation_status.get('errors'):
            self.console.print("\n[red]Errors:[/red]")
            for error in validation_status['errors']:
                self.console.print(f"  • {error}")
        
        if validation_status.get('warnings'):
            self.console.print("\n[yellow]Warnings:[/yellow]")
            for warning in validation_status['warnings']:
                self.console.print(f"  • {warning}")
    
    async def _validate_improvements(self):
        """Validate staged improvements."""
        staged = self.improver.get_staged_improvements('staged')
        
        if not staged:
            self.console.print("[yellow]No staged improvements to validate[/yellow]")
            return
        
        self.console.print(f"\n[bold]Validating {len(staged)} improvements...[/bold]")
        
        results = {}
        with self.console.status("[bold green]Validating...") as status:
            for imp in staged:
                status.update(f"Validating {os.path.basename(imp.modification.target_file)}...")
                result = await self.improver.validate_staged_improvement(
                    imp.metadata['staging_id']
                )
                results[imp.metadata['staging_id']] = result
        
        # Show summary
        ready_count = sum(1 for r in results.values() if r.get('ready_to_apply'))
        self.console.print(f"\n✅ {ready_count}/{len(results)} improvements ready to apply")
        
        # Show details for failed validations
        for staging_id, result in results.items():
            if not result.get('ready_to_apply'):
                self.console.print(f"\n[red]Failed:[/red] {staging_id[:12]}...")
                if result.get('errors'):
                    for error in result['errors'][:3]:
                        self.console.print(f"  • {error}")
    
    async def _validate_single(self, staging_id: str):
        """Validate a single improvement."""
        with self.console.status("[bold green]Validating..."):
            result = await self.improver.validate_staged_improvement(staging_id)
        
        self._show_validation_status(result)
    
    async def _apply_improvements(self):
        """Apply validated improvements."""
        validated = self.improver.get_staged_improvements('validated')
        
        if not validated:
            self.console.print("[yellow]No validated improvements to apply[/yellow]")
            return
        
        # Show list
        self.console.print(f"\n[bold]Ready to apply {len(validated)} improvements:[/bold]")
        for imp in validated:
            self.console.print(
                f"  • {imp.metadata['staging_id'][:12]}... - "
                f"{os.path.basename(imp.modification.target_file)}"
            )
        
        if not Confirm.ask("\n[yellow]Apply all validated improvements?[/yellow]"):
            return
        
        # Apply improvements
        success_count = 0
        for imp in track(validated, description="Applying improvements..."):
            # Start monitoring
            metrics = await self.monitor.start_monitoring(
                imp.metadata['staging_id'],
                imp.modification.target_file
            )
            
            # Apply improvement
            success = await self.improver.apply_staged_improvement(
                imp.metadata['staging_id']
            )
            
            # Stop monitoring
            await self.monitor.stop_monitoring(
                imp.metadata['staging_id'],
                improvement_applied=success
            )
            
            if success:
                success_count += 1
        
        self.console.print(
            f"\n✅ Successfully applied {success_count}/{len(validated)} improvements"
        )
    
    async def _apply_single(self, staging_id: str):
        """Apply a single improvement."""
        if not Confirm.ask(f"\n[yellow]Apply improvement {staging_id[:12]}...?[/yellow]"):
            return
        
        # Start monitoring
        imp = self.improver.staged_improvements[staging_id]
        metrics = await self.monitor.start_monitoring(
            staging_id,
            imp.modification.target_file
        )
        
        # Apply
        with self.console.status("[bold green]Applying improvement..."):
            success = await self.improver.apply_staged_improvement(staging_id)
        
        # Stop monitoring
        final_metrics = await self.monitor.stop_monitoring(
            staging_id,
            improvement_applied=success
        )
        
        if success:
            self.console.print(f"\n✅ Successfully applied improvement")
            self.console.print(f"Verdict: {final_metrics.verdict}")
        else:
            self.console.print(f"\n❌ Failed to apply improvement")
    
    def _reject_improvement(self, staging_id: str):
        """Reject an improvement."""
        if not Confirm.ask(f"\n[red]Reject improvement {staging_id[:12]}...?[/red]"):
            return
        
        # Move to rejected directory (or delete)
        imp = self.improver.staged_improvements[staging_id]
        staging_dir = os.path.dirname(imp.staged_path)
        
        import shutil
        shutil.rmtree(staging_dir)
        del self.improver.staged_improvements[staging_id]
        
        self.console.print("[green]Improvement rejected[/green]")
    
    def _view_monitoring_reports(self):
        """View monitoring reports."""
        reports_dir = os.path.join(self.repo_path, '.self_improver', 'reports')
        
        if not os.path.exists(reports_dir):
            self.console.print("[yellow]No monitoring reports found[/yellow]")
            return
        
        # List reports
        reports = [f for f in os.listdir(reports_dir) if f.startswith('monitoring_')]
        
        if not reports:
            self.console.print("[yellow]No monitoring reports found[/yellow]")
            return
        
        self.console.print(f"\n[bold]Available Reports ({len(reports)}):[/bold]")
        for i, report in enumerate(reports[-10:], 1):  # Show last 10
            self.console.print(f"{i}. {report}")
        
        choice = Prompt.ask("\nSelect report number (or 'back')")
        
        if choice.lower() == 'back':
            return
        
        try:
            report_file = reports[-10:][int(choice) - 1]
            report_path = os.path.join(reports_dir, report_file)
            
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            # Display report
            self.console.print(f"\n[bold]Monitoring Report:[/bold]")
            self.console.print(f"Staging ID: [cyan]{report_data['staging_id']}[/cyan]")
            self.console.print(f"File: [green]{report_data['file_path']}[/green]")
            self.console.print(f"Verdict: [yellow]{report_data['verdict'].upper()}[/yellow]")
            self.console.print(f"Duration: {report_data['monitoring_duration']}")
            
            if report_data['improvements']:
                self.console.print("\n[green]Improvements:[/green]")
                for imp in report_data['improvements']:
                    self.console.print(
                        f"  • {imp['metric']}: "
                        f"{imp['before']} → {imp['after']} "
                        f"({imp['change_percent']:+.1f}%)"
                    )
            
            if report_data['degradations']:
                self.console.print("\n[red]Degradations:[/red]")
                for deg in report_data['degradations']:
                    self.console.print(
                        f"  • {deg['metric']}: "
                        f"{deg['before']} → {deg['after']} "
                        f"({deg['change_percent']:+.1f}%)"
                    )
                    
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/red]")
    
    def _generate_summary_report(self):
        """Generate and display summary report."""
        report = self.improver.generate_staging_report()
        
        self.console.print("\n[bold]Staging Summary Report:[/bold]")
        self.console.print(f"Generated: {report['timestamp']}")
        
        # Summary table
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")
        
        summary_table.add_row("Total Staged", str(report['summary']['total_staged']))
        summary_table.add_row("Total Validated", str(report['summary']['total_validated']))
        summary_table.add_row("Total Applied", str(report['summary']['total_applied']))
        summary_table.add_row(
            "Success Rate", 
            f"{report['summary']['success_rate']:.1%}"
        )
        
        self.console.print(summary_table)
        
        # By type
        if report['by_type']:
            self.console.print("\n[bold]By Type:[/bold]")
            type_table = Table(show_header=True, header_style="bold blue")
            type_table.add_column("Type", style="yellow")
            type_table.add_column("Count", style="white")
            type_table.add_column("Applied", style="green")
            
            for imp_type, data in report['by_type'].items():
                type_table.add_row(
                    imp_type,
                    str(data['count']),
                    str(data['applied'])
                )
            
            self.console.print(type_table)
        
        # Recent activity
        if report['recent_activity']:
            self.console.print("\n[bold]Recent Activity:[/bold]")
            for activity in report['recent_activity'][:5]:
                status_color = {
                    'applied': 'green',
                    'validated': 'yellow',
                    'staged': 'blue'
                }.get(activity['status'], 'white')
                
                self.console.print(
                    f"  • [{status_color}]{activity['status'].upper()}[/{status_color}] "
                    f"{activity['type']} - {activity['file']} "
                    f"({activity['created_at'][:10]})"
                )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Review and manage staged code improvements"
    )
    parser.add_argument(
        '--repo',
        default='.',
        help='Repository path (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Run reviewer
    reviewer = ImprovementReviewer(args.repo)
    reviewer.run()


if __name__ == "__main__":
    main()
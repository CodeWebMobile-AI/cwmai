#!/usr/bin/env python3
"""
AI API Log Viewer

Real-time viewer for AI API communication logs with filtering and statistics.
"""

import json
import argparse
import sys
import time
import os
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Any, List, Optional
import signal


class AIAPILogViewer:
    """Interactive viewer for AI API communication logs."""
    
    def __init__(self, log_file: str = "ai_api_communication.log"):
        """Initialize the log viewer."""
        self.log_file = log_file
        self.stats = defaultdict(int)
        self.provider_stats = defaultdict(lambda: {"requests": 0, "errors": 0, "cache_hits": 0, "total_time": 0.0})
        self.model_stats = defaultdict(lambda: {"requests": 0, "errors": 0, "total_cost": 0.0})
        self.hourly_stats = defaultdict(int)
        self.running = True
        
    def parse_log_entry(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON log entry."""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None
    
    def update_stats(self, entry: Dict[str, Any]):
        """Update statistics from a log entry."""
        event_type = entry.get('event_type', '')
        
        if event_type == 'request_start':
            metadata = entry.get('request_metadata', {})
            provider = metadata.get('provider', 'unknown')
            model = metadata.get('model', 'unknown')
            
            self.stats['total_requests'] += 1
            self.provider_stats[provider]['requests'] += 1
            self.model_stats[model]['requests'] += 1
            
            # Track hourly
            timestamp = metadata.get('timestamp', '')
            if timestamp:
                hour = timestamp[:13]  # YYYY-MM-DDTHH
                self.hourly_stats[hour] += 1
        
        elif event_type == 'request_complete':
            metadata = entry.get('response_metadata', {})
            provider = entry.get('provider', 'unknown')
            model = entry.get('model', 'unknown')
            
            response_time = metadata.get('response_time', 0)
            cost = metadata.get('cost_estimate', 0)
            
            self.stats['total_time'] += response_time
            self.provider_stats[provider]['total_time'] += response_time
            self.model_stats[model]['total_cost'] += cost
            
            if metadata.get('cached'):
                self.stats['cache_hits'] += 1
                self.provider_stats[provider]['cache_hits'] += 1
        
        elif event_type == 'request_error':
            provider = entry.get('provider', 'unknown')
            self.stats['total_errors'] += 1
            self.provider_stats[provider]['errors'] += 1
        
        elif event_type == 'cache_hit':
            self.stats['cache_events'] += 1
    
    def format_entry(self, entry: Dict[str, Any], verbose: bool = False) -> str:
        """Format a log entry for display."""
        event_type = entry.get('event_type', 'unknown')
        timestamp = entry.get('timestamp', entry.get('request_metadata', {}).get('timestamp', ''))
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S.%f')[:-3]
            except:
                time_str = timestamp[:19]
        else:
            time_str = "??:??:??"
        
        if event_type == 'request_start':
            metadata = entry.get('request_metadata', {})
            provider = metadata.get('provider', 'unknown')
            model = metadata.get('model', 'unknown')
            prompt_len = metadata.get('prompt_length', 0)
            request_id = metadata.get('request_id', 'unknown')
            
            line = f"[{time_str}] 🚀 REQUEST START [{request_id}] {provider}/{model} ({prompt_len} chars)"
            
            if metadata.get('round_robin'):
                line += " [ROUND-ROBIN]"
            if metadata.get('distributed'):
                line += " [DISTRIBUTED]"
            
            if verbose and 'prompt_preview' in entry:
                line += f"\n    Prompt: {entry['prompt_preview'][:100]}..."
                
        elif event_type == 'request_complete':
            request_id = entry.get('request_id', 'unknown')
            metadata = entry.get('response_metadata', {})
            provider = entry.get('provider', 'unknown')
            model = entry.get('model', 'unknown')
            
            response_time = metadata.get('response_time', 0)
            response_len = metadata.get('response_length', 0)
            cached = metadata.get('cached', False)
            cost = metadata.get('cost_estimate', 0)
            
            cache_str = " 💾 CACHED" if cached else ""
            line = f"[{time_str}] ✅ REQUEST COMPLETE [{request_id}] {provider}/{model} " \
                   f"({response_time:.2f}s, {response_len} chars, ${cost:.4f}){cache_str}"
            
            if verbose and 'response_preview' in entry:
                line += f"\n    Response: {entry['response_preview'][:100]}..."
                
        elif event_type == 'request_error':
            request_id = entry.get('request_id', 'unknown')
            provider = entry.get('provider', 'unknown')
            error_type = entry.get('error_type', 'unknown')
            error_msg = entry.get('error_message', 'unknown')
            
            line = f"[{time_str}] ❌ REQUEST ERROR [{request_id}] {provider} - {error_type}: {error_msg}"
            
        elif event_type == 'cache_hit':
            request_id = entry.get('request_id', 'unknown')
            provider = entry.get('provider', 'unknown')
            backend = entry.get('cache_backend', 'unknown')
            
            line = f"[{time_str}] 💾 CACHE HIT [{request_id}] {provider} (backend: {backend})"
            
        elif event_type == 'retry_attempt':
            request_id = entry.get('request_id', 'unknown')
            provider = entry.get('provider', 'unknown')
            attempt = entry.get('attempt', 0)
            reason = entry.get('reason', 'unknown')
            
            line = f"[{time_str}] 🔄 RETRY [{request_id}] {provider} attempt {attempt} - {reason}"
            
        else:
            line = f"[{time_str}] {event_type}: {json.dumps(entry)[:100]}..."
        
        return line
    
    def print_statistics(self):
        """Print current statistics."""
        print("\n" + "="*80)
        print("📊 AI API COMMUNICATION STATISTICS")
        print("="*80)
        
        total_requests = self.stats['total_requests']
        if total_requests == 0:
            print("No requests logged yet.")
            return
        
        avg_time = self.stats['total_time'] / total_requests if total_requests > 0 else 0
        cache_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        error_rate = (self.stats['total_errors'] / total_requests * 100) if total_requests > 0 else 0
        
        print(f"\n📈 Overall Stats:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Errors: {self.stats['total_errors']} ({error_rate:.1f}%)")
        print(f"  Cache Hits: {self.stats['cache_hits']} ({cache_rate:.1f}%)")
        print(f"  Average Response Time: {avg_time:.3f}s")
        
        print(f"\n🔧 Provider Stats:")
        for provider, stats in sorted(self.provider_stats.items()):
            if stats['requests'] > 0:
                avg_provider_time = stats['total_time'] / stats['requests']
                provider_error_rate = (stats['errors'] / stats['requests'] * 100)
                provider_cache_rate = (stats['cache_hits'] / stats['requests'] * 100)
                
                print(f"  {provider}:")
                print(f"    Requests: {stats['requests']}")
                print(f"    Errors: {stats['errors']} ({provider_error_rate:.1f}%)")
                print(f"    Cache Hits: {stats['cache_hits']} ({provider_cache_rate:.1f}%)")
                print(f"    Avg Time: {avg_provider_time:.3f}s")
        
        print(f"\n🤖 Model Stats:")
        for model, stats in sorted(self.model_stats.items()):
            if stats['requests'] > 0:
                print(f"  {model}:")
                print(f"    Requests: {stats['requests']}")
                print(f"    Total Cost: ${stats['total_cost']:.4f}")
                print(f"    Avg Cost: ${stats['total_cost']/stats['requests']:.4f}")
        
        # Show recent activity
        if self.hourly_stats:
            print(f"\n📅 Recent Activity (last 24 hours):")
            recent_hours = sorted(self.hourly_stats.items())[-24:]
            for hour, count in recent_hours[-5:]:  # Show last 5 hours
                print(f"  {hour}: {count} requests")
        
        print("="*80)
    
    def tail_logs(self, follow: bool = True, filter_provider: Optional[str] = None,
                  filter_event: Optional[str] = None, verbose: bool = False):
        """Tail the log file with optional filtering."""
        try:
            # Start from end of file if following
            if follow and os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    # Go to end of file
                    f.seek(0, 2)
                    position = f.tell()
            else:
                position = 0
            
            print(f"📡 Monitoring AI API communications from {self.log_file}")
            print(f"Filters: provider={filter_provider or 'all'}, event={filter_event or 'all'}")
            print("Press Ctrl+C to stop and show statistics\n")
            
            while self.running:
                try:
                    with open(self.log_file, 'r') as f:
                        f.seek(position)
                        
                        for line in f:
                            if not line.strip():
                                continue
                            
                            entry = self.parse_log_entry(line)
                            if not entry:
                                continue
                            
                            # Apply filters
                            if filter_provider:
                                provider = entry.get('provider') or \
                                          entry.get('request_metadata', {}).get('provider')
                                if provider != filter_provider:
                                    continue
                            
                            if filter_event and entry.get('event_type') != filter_event:
                                continue
                            
                            # Update statistics
                            self.update_stats(entry)
                            
                            # Format and print
                            formatted = self.format_entry(entry, verbose)
                            print(formatted)
                        
                        position = f.tell()
                    
                    if follow:
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                    else:
                        break
                        
                except FileNotFoundError:
                    if not follow:
                        print(f"Log file {self.log_file} not found.")
                        break
                    time.sleep(1)  # Wait for file to be created
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped monitoring.")
    
    def run(self, args):
        """Run the log viewer with given arguments."""
        # Set up signal handler
        signal.signal(signal.SIGINT, lambda sig, frame: setattr(self, 'running', False))
        
        # Tail logs
        self.tail_logs(
            follow=args.follow,
            filter_provider=args.provider,
            filter_event=args.event,
            verbose=args.verbose
        )
        
        # Show statistics
        if args.stats or not args.no_stats:
            self.print_statistics()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time viewer for AI API communication logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Follow logs in real-time
  python ai_api_log_viewer.py -f
  
  # Show only errors
  python ai_api_log_viewer.py --event request_error
  
  # Filter by provider
  python ai_api_log_viewer.py --provider anthropic
  
  # Verbose mode with full content
  python ai_api_log_viewer.py -f -v
  
  # Just show statistics
  python ai_api_log_viewer.py --stats --no-follow
        """
    )
    
    parser.add_argument(
        '-f', '--follow',
        action='store_true',
        help='Follow log file in real-time (like tail -f)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output including content previews'
    )
    
    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openai', 'gemini', 'deepseek'],
        help='Filter by AI provider'
    )
    
    parser.add_argument(
        '--event',
        choices=['request_start', 'request_complete', 'request_error', 
                 'cache_hit', 'cache_miss', 'cache_store', 'retry_attempt'],
        help='Filter by event type'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Do not show statistics at the end'
    )
    
    parser.add_argument(
        '--log-file',
        default='ai_api_communication.log',
        help='Path to log file (default: ai_api_communication.log)'
    )
    
    args = parser.parse_args()
    
    # Create viewer and run
    viewer = AIAPILogViewer(args.log_file)
    viewer.run(args)


if __name__ == "__main__":
    main()
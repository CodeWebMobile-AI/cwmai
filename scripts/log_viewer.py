#!/usr/bin/env python3
"""
Enhanced log viewer with filtering and formatting capabilities.
"""
import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple
import os


class LogAnalyzer:
    """Analyzes and formats log files for better readability."""
    
    def __init__(self):
        self.patterns = {
            'timestamp': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})',
            'level': r' - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - ',
            'module': r' - ([^-]+) - (?:DEBUG|INFO|WARNING|ERROR|CRITICAL) - ',
            'worker': r'worker_\d+',
            'uuid': r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
        }
        
        # Color codes for terminal output
        self.colors = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'RESET': '\033[0m',
            'BOLD': '\033[1m',
            'DIM': '\033[2m'
        }
        
    def parse_line(self, line: str) -> Dict[str, str]:
        """Parse a log line into components."""
        result = {'raw': line}
        
        # Extract timestamp
        timestamp_match = re.search(self.patterns['timestamp'], line)
        if timestamp_match:
            result['timestamp'] = timestamp_match.group(1)
            
        # Extract level
        level_match = re.search(self.patterns['level'], line)
        if level_match:
            result['level'] = level_match.group(1)
            
        # Extract module
        module_match = re.search(self.patterns['module'], line)
        if module_match:
            result['module'] = module_match.group(1)
            
        # Extract message
        if 'timestamp' in result and 'module' in result and 'level' in result:
            prefix_pattern = f"{re.escape(result['timestamp'])} - {re.escape(result['module'])} - {result['level']} - "
            message_start = re.search(prefix_pattern, line)
            if message_start:
                result['message'] = line[message_start.end():]
            else:
                result['message'] = line
                
        return result
    
    def format_line(self, parsed: Dict[str, str], use_colors: bool = True) -> str:
        """Format a parsed log line for display."""
        if 'level' not in parsed:
            return parsed['raw']
            
        level = parsed['level']
        
        if use_colors:
            level_color = self.colors.get(level, '')
            reset = self.colors['RESET']
            dim = self.colors['DIM']
            bold = self.colors['BOLD']
        else:
            level_color = reset = dim = bold = ''
            
        # Format timestamp
        timestamp = parsed.get('timestamp', '')
        if timestamp:
            # Simplify timestamp - remove milliseconds for readability
            try:
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                timestamp = dt.strftime('%H:%M:%S')
            except:
                timestamp = timestamp.split(',')[0].split(' ')[1]
                
        # Format module name
        module = parsed.get('module', '').ljust(25)[:25]
        
        # Format level
        level_str = level.ljust(8)
        
        # Format message
        message = parsed.get('message', '')
        
        # Shorten UUIDs in message
        message = re.sub(self.patterns['uuid'], lambda m: m.group(0)[:8] + '...', message)
        
        return f"{dim}{timestamp}{reset} {level_color}{bold}{level_str}{reset} {module} {message}"
    
    def analyze_frequency(self, lines: List[str]) -> Dict[str, int]:
        """Analyze message frequency."""
        messages = defaultdict(int)
        
        for line in lines:
            parsed = self.parse_line(line)
            if 'message' in parsed:
                # Normalize message
                msg = re.sub(r'\d+', 'N', parsed['message'])
                msg = re.sub(self.patterns['uuid'], 'UUID', msg)
                msg = re.sub(self.patterns['worker'], 'worker_N', msg)
                messages[msg] += 1
                
        return dict(sorted(messages.items(), key=lambda x: x[1], reverse=True))
    
    def filter_lines(self, lines: List[str], 
                    level: str = None,
                    module: str = None,
                    message: str = None,
                    exclude_pattern: str = None) -> List[str]:
        """Filter log lines based on criteria."""
        filtered = []
        
        for line in lines:
            parsed = self.parse_line(line)
            
            # Level filter
            if level and parsed.get('level') != level.upper():
                continue
                
            # Module filter
            if module and module.lower() not in parsed.get('module', '').lower():
                continue
                
            # Message filter
            if message and message.lower() not in parsed.get('message', '').lower():
                continue
                
            # Exclude pattern
            if exclude_pattern and re.search(exclude_pattern, line):
                continue
                
            filtered.append(line)
            
        return filtered


def main():
    parser = argparse.ArgumentParser(description='Enhanced log viewer with filtering and formatting')
    parser.add_argument('logfile', nargs='?', default='continuous_ai.log',
                      help='Log file to analyze (default: continuous_ai.log)')
    parser.add_argument('-n', '--lines', type=int, default=100,
                      help='Number of lines to display (default: 100, use -1 for all)')
    parser.add_argument('-l', '--level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                      help='Filter by log level')
    parser.add_argument('-m', '--module', help='Filter by module name')
    parser.add_argument('-s', '--search', help='Search for message content')
    parser.add_argument('-x', '--exclude', help='Exclude lines matching regex pattern')
    parser.add_argument('-f', '--follow', action='store_true',
                      help='Follow log file (like tail -f)')
    parser.add_argument('--no-color', action='store_true',
                      help='Disable colored output')
    parser.add_argument('--frequency', action='store_true',
                      help='Show message frequency analysis')
    parser.add_argument('--tail', action='store_true',
                      help='Show last N lines (default behavior)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.logfile):
        print(f"Error: Log file '{args.logfile}' not found", file=sys.stderr)
        sys.exit(1)
        
    analyzer = LogAnalyzer()
    
    # Read log file
    with open(args.logfile, 'r') as f:
        all_lines = f.readlines()
        
    # Filter lines
    filtered_lines = analyzer.filter_lines(
        all_lines,
        level=args.level,
        module=args.module,
        message=args.search,
        exclude_pattern=args.exclude
    )
    
    if args.frequency:
        # Show frequency analysis
        print("\n=== Message Frequency Analysis ===\n")
        frequencies = analyzer.analyze_frequency(filtered_lines)
        for msg, count in list(frequencies.items())[:20]:
            if count > 1:
                print(f"{count:5d}x: {msg[:100]}")
        return
        
    # Select lines to display
    if args.lines == -1:
        display_lines = filtered_lines
    else:
        display_lines = filtered_lines[-args.lines:]
        
    # Display lines
    for line in display_lines:
        parsed = analyzer.parse_line(line.strip())
        formatted = analyzer.format_line(parsed, use_colors=not args.no_color)
        print(formatted)
        
    if args.follow:
        # Follow mode
        print("\n--- Following log file (Ctrl+C to exit) ---\n")
        try:
            import time
            last_size = os.path.getsize(args.logfile)
            while True:
                time.sleep(0.5)
                current_size = os.path.getsize(args.logfile)
                if current_size > last_size:
                    with open(args.logfile, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                        for line in new_lines:
                            parsed = analyzer.parse_line(line.strip())
                            if args.level and parsed.get('level') != args.level.upper():
                                continue
                            if args.module and args.module.lower() not in parsed.get('module', '').lower():
                                continue
                            if args.search and args.search.lower() not in parsed.get('message', '').lower():
                                continue
                            if args.exclude and re.search(args.exclude, line):
                                continue
                            formatted = analyzer.format_line(parsed, use_colors=not args.no_color)
                            print(formatted)
                    last_size = current_size
        except KeyboardInterrupt:
            print("\n\nExiting...")


if __name__ == '__main__':
    main()
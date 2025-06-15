#!/usr/bin/env python3
"""
Log file cleaner that deduplicates and reformats existing log files.
"""
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import argparse
import os
import shutil


class LogCleaner:
    """Cleans and deduplicates log files."""
    
    def __init__(self, time_window_seconds: int = 5):
        self.time_window = timedelta(seconds=time_window_seconds)
        self.timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
        
    def parse_timestamp(self, line: str) -> datetime:
        """Extract timestamp from log line."""
        match = self.timestamp_pattern.search(line)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            except:
                pass
        return datetime.now()
    
    def normalize_message(self, line: str) -> str:
        """Normalize a log line for comparison."""
        # Remove timestamp
        normalized = self.timestamp_pattern.sub('', line)
        
        # Remove UUIDs
        normalized = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', normalized)
        
        # Normalize worker IDs
        normalized = re.sub(r'worker_\d+', 'worker_N', normalized)
        
        # Normalize numbers (but preserve log levels)
        normalized = re.sub(r'(?<![\w])(\d+\.\d+)(?![\w])', 'X.X', normalized)
        normalized = re.sub(r'(?<![\w])(\d+)(?![\w])', 'N', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def clean_log_file(self, input_file: str, output_file: str = None, 
                      min_count: int = 2, preserve_singles: bool = True) -> Dict[str, int]:
        """
        Clean and deduplicate a log file.
        
        Args:
            input_file: Path to input log file
            output_file: Path to output file (if None, overwrites input)
            min_count: Minimum count to show deduplication
            preserve_singles: Whether to keep single occurrences
            
        Returns:
            Statistics about the cleaning
        """
        if output_file is None:
            output_file = input_file + '.tmp'
            overwrite = True
        else:
            overwrite = False
            
        stats = {
            'total_lines': 0,
            'output_lines': 0,
            'grouped_messages': 0,
            'total_duplicates': 0
        }
        
        print(f"📖 Reading {input_file}...")
        
        # Group messages by normalized form and time window
        message_groups = defaultdict(list)
        
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                    
                stats['total_lines'] += 1
                timestamp = self.parse_timestamp(line)
                normalized = self.normalize_message(line)
                
                message_groups[normalized].append((timestamp, line))
        
        print(f"✅ Read {stats['total_lines']} lines")
        print(f"📊 Found {len(message_groups)} unique message patterns")
        
        # Sort all messages by timestamp
        all_messages = []
        for normalized, occurrences in message_groups.items():
            all_messages.extend(occurrences)
        all_messages.sort(key=lambda x: x[0])
        
        # Write deduplicated output
        print(f"✍️  Writing cleaned log...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            last_written = {}
            pending_groups = defaultdict(list)
            
            for timestamp, line in all_messages:
                normalized = self.normalize_message(line)
                
                # Check if we should group this message
                last_time = last_written.get(normalized)
                
                if last_time and timestamp - last_time <= self.time_window:
                    # Add to pending group
                    pending_groups[normalized].append((timestamp, line))
                else:
                    # Write any pending group for this normalized message
                    if normalized in pending_groups:
                        group = pending_groups[normalized]
                        if len(group) >= min_count:
                            # Write first message with count
                            first_msg = group[0][1]
                            f.write(f"{first_msg} [×{len(group)}]\n")
                            stats['output_lines'] += 1
                            stats['grouped_messages'] += 1
                            stats['total_duplicates'] += len(group) - 1
                        else:
                            # Write individual messages
                            for _, msg in group:
                                if preserve_singles or len(group) > 1:
                                    f.write(f"{msg}\n")
                                    stats['output_lines'] += 1
                        
                        del pending_groups[normalized]
                    
                    # Start new group or write single
                    pending_groups[normalized] = [(timestamp, line)]
                    last_written[normalized] = timestamp
            
            # Write remaining pending groups
            for normalized, group in pending_groups.items():
                if len(group) >= min_count:
                    first_msg = group[0][1]
                    f.write(f"{first_msg} [×{len(group)}]\n")
                    stats['output_lines'] += 1
                    stats['grouped_messages'] += 1
                    stats['total_duplicates'] += len(group) - 1
                else:
                    for _, msg in group:
                        if preserve_singles or len(group) > 1:
                            f.write(f"{msg}\n")
                            stats['output_lines'] += 1
        
        # Overwrite original if needed
        if overwrite:
            # Create backup
            backup_file = input_file + '.bak'
            shutil.copy2(input_file, backup_file)
            print(f"💾 Created backup: {backup_file}")
            
            # Replace original
            shutil.move(output_file, input_file)
            print(f"✅ Updated {input_file}")
        else:
            print(f"✅ Created {output_file}")
            
        return stats


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description='Clean and deduplicate log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean continuous_ai.log with default settings
  python scripts/clean_log_file.py
  
  # Clean with custom time window
  python scripts/clean_log_file.py -w 10
  
  # Save to new file instead of overwriting
  python scripts/clean_log_file.py -o cleaned.log
  
  # Show only messages that repeat 5+ times
  python scripts/clean_log_file.py -m 5
  
  # Remove single occurrences entirely
  python scripts/clean_log_file.py --no-singles
""")
    
    parser.add_argument('logfile', nargs='?', default='continuous_ai.log',
                      help='Log file to clean (default: continuous_ai.log)')
    parser.add_argument('-o', '--output', 
                      help='Output file (default: overwrite input file)')
    parser.add_argument('-w', '--window', type=int, default=5,
                      help='Time window in seconds for grouping (default: 5)')
    parser.add_argument('-m', '--min-count', type=int, default=2,
                      help='Minimum count to show deduplication (default: 2)')
    parser.add_argument('--no-singles', action='store_true',
                      help='Remove single occurrences from output')
    parser.add_argument('--no-backup', action='store_true',
                      help='Do not create backup when overwriting')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.logfile):
        print(f"❌ Error: Log file '{args.logfile}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Get file sizes
    original_size = os.path.getsize(args.logfile)
    
    print(f"🧹 Cleaning {args.logfile}")
    print(f"📏 Original size: {format_size(original_size)}")
    print()
    
    cleaner = LogCleaner(time_window_seconds=args.window)
    
    try:
        stats = cleaner.clean_log_file(
            args.logfile,
            args.output,
            min_count=args.min_count,
            preserve_singles=not args.no_singles
        )
        
        # Get new file size
        output_file = args.output or args.logfile
        new_size = os.path.getsize(output_file)
        
        print()
        print("📊 Cleaning Statistics:")
        print(f"   Original lines: {stats['total_lines']:,}")
        print(f"   Output lines:   {stats['output_lines']:,}")
        print(f"   Grouped msgs:   {stats['grouped_messages']:,}")
        print(f"   Duplicates:     {stats['total_duplicates']:,}")
        print(f"   Reduction:      {stats['total_lines'] - stats['output_lines']:,} lines")
        print()
        print(f"💾 File size reduction:")
        print(f"   Original: {format_size(original_size)}")
        print(f"   New:      {format_size(new_size)}")
        print(f"   Saved:    {format_size(original_size - new_size)} ({(1 - new_size/original_size)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
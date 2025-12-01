"""
Analyze monitor logs to identify root cause of run.py stopping
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_logs(log_file):
    """Load all log entries from JSONL file"""
    entries = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return entries

def analyze_logs(entries):
    """Analyze logs to find root causes"""
    if not entries:
        print("No log entries found!")
        return
    
    print(f"Analyzing {len(entries)} log entries...\n")
    
    # Track key metrics over time
    memory_peaks = []
    cpu_peaks = []
    disk_peaks = []
    swap_peaks = []
    process_missing = []
    vm_restarts = []
    alerts_timeline = []
    
    for entry in entries:
        timestamp = entry['timestamp']
        metrics = entry['metrics']
        alerts = entry.get('alerts', [])
        
        # Memory analysis
        mem = metrics['memory']
        memory_peaks.append((timestamp, mem['percent'], mem['used_gb'], mem['available_gb']))
        swap_peaks.append((timestamp, mem['swap_percent'], mem['swap_used_gb']))
        
        # CPU analysis
        cpu = metrics['cpu']
        cpu_peaks.append((timestamp, cpu['percent']))
        
        # Disk analysis
        disk = metrics['disk']
        disk_peaks.append((timestamp, disk['percent'], disk['free_gb']))
        
        # Process status
        process = metrics.get('process', {})
        if process.get('status') == 'not_found' or 'error' in process:
            process_missing.append(timestamp)
        
        # VM restarts
        if metrics.get('vm_restarted'):
            vm_restarts.append(timestamp)
        
        # Alerts
        if alerts:
            alerts_timeline.append((timestamp, alerts))
    
    # Find maximums
    max_memory = max(memory_peaks, key=lambda x: x[1])
    max_cpu = max(cpu_peaks, key=lambda x: x[1])
    max_disk = max(disk_peaks, key=lambda x: x[1])
    max_swap = max(swap_peaks, key=lambda x: x[1])
    
    print("="*70)
    print("ROOT CAUSE ANALYSIS")
    print("="*70)
    
    # Memory analysis
    print(f"\n📊 MEMORY ANALYSIS:")
    print(f"  Peak Memory Usage: {max_memory[1]:.1f}% at {max_memory[0]}")
    print(f"    Used: {max_memory[2]:.2f}GB, Available: {max_memory[3]:.2f}GB")
    
    if max_swap[1] > 10:
        print(f"  ⚠️  HIGH SWAP USAGE: {max_swap[1]:.1f}% ({max_swap[2]:.2f}GB) at {max_swap[0]}")
        print(f"     This indicates memory pressure - system was using swap!")
    
    # Check if memory was consistently high before process stopped
    last_entries = entries[-10:] if len(entries) >= 10 else entries
    avg_memory_before_stop = sum(e['metrics']['memory']['percent'] for e in last_entries) / len(last_entries)
    if avg_memory_before_stop > 85:
        print(f"  🚨 MEMORY PRESSURE: Average {avg_memory_before_stop:.1f}% in last {len(last_entries)} checks")
        print(f"     Likely cause: Out of Memory (OOM) killed the process")
    
    # CPU analysis
    print(f"\n📊 CPU ANALYSIS:")
    print(f"  Peak CPU Usage: {max_cpu[1]:.1f}% at {max_cpu[0]}")
    avg_cpu = sum(cpu[1] for cpu in cpu_peaks) / len(cpu_peaks)
    print(f"  Average CPU Usage: {avg_cpu:.1f}%")
    
    # Disk analysis
    print(f"\n📊 DISK ANALYSIS:")
    print(f"  Peak Disk Usage: {max_disk[1]:.1f}% at {max_disk[0]}")
    print(f"  Free Space: {max_disk[2]:.2f}GB")
    if max_disk[1] > 90:
        print(f"  ⚠️  DISK SPACE CRITICAL: {max_disk[1]:.1f}% full")
    
    # Process status
    print(f"\n📊 PROCESS STATUS:")
    if process_missing:
        print(f"  ⚠️  Process missing at {len(process_missing)} check(s)")
        print(f"  First missing: {process_missing[0]}")
        print(f"  Last missing: {process_missing[-1]}")
    else:
        print(f"  ✓ Process was running throughout monitoring")
    
    # VM restarts
    print(f"\n📊 VM RESTART ANALYSIS:")
    if vm_restarts:
        print(f"  🚨 VM RESTARTED {len(vm_restarts)} time(s):")
        for restart_time in vm_restarts:
            print(f"    - {restart_time}")
        print(f"  This is likely the cause of run.py stopping!")
    else:
        print(f"  ✓ No VM restarts detected")
    
    # Alert timeline
    print(f"\n📊 ALERT TIMELINE:")
    if alerts_timeline:
        print(f"  Found {len(alerts_timeline)} alert events:")
        for timestamp, alerts in alerts_timeline[-10:]:  # Show last 10
            print(f"    [{timestamp}]")
            for alert in alerts:
                print(f"      - {alert}")
    else:
        print(f"  ✓ No alerts triggered")
    
    # Final diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    print(f"{'='*70}")
    
    causes = []
    
    if vm_restarts:
        causes.append("🚨 PRIMARY CAUSE: VM RESTART - The VM was restarted, which stopped run.py")
    
    if avg_memory_before_stop > 85 or max_memory[1] > 95:
        causes.append("🚨 PRIMARY CAUSE: OUT OF MEMORY - High memory usage likely caused OOM killer to terminate run.py")
    
    if max_swap[1] > 50:
        causes.append("⚠️  CONTRIBUTING FACTOR: Heavy swap usage indicates memory pressure")
    
    if max_disk[1] > 95:
        causes.append("🚨 PRIMARY CAUSE: DISK FULL - No space left, process may have crashed")
    
    if process_missing and not vm_restarts:
        if avg_memory_before_stop > 80:
            causes.append("🚨 LIKELY CAUSE: Process killed by OOM killer due to memory exhaustion")
        else:
            causes.append("⚠️  Process stopped unexpectedly - check run.py logs for errors")
    
    if not causes:
        causes.append("✓ No obvious issues detected in system metrics")
        causes.append("  Check run.py logs for application-level errors")
    
    for cause in causes:
        print(f"  {cause}")
    
    print(f"\n{'='*70}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze monitor logs")
    parser.add_argument('--log-file', type=str, default='monitor_log.jsonl',
                       help='Log file to analyze (default: monitor_log.jsonl)')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    entries = load_logs(log_path)
    analyze_logs(entries)

if __name__ == "__main__":
    main()


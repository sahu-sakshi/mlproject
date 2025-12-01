"""
Monitoring script for run.py training process
Tracks: Memory, CPU, VM restarts, Process status, Disk space
"""
import psutil
import time
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import subprocess
import platform

class SystemMonitor:
    def __init__(self, log_file="monitor_log.jsonl", check_interval=30, alert_thresholds=None):
        self.log_file = Path(log_file)
        self.check_interval = check_interval
        self.alert_thresholds = alert_thresholds or {
            'memory_percent': 90,
            'cpu_percent': 95,
            'disk_percent': 90
        }
        self.start_time = time.time()
        self.last_boot_time = self.get_boot_time()
        self.process_pid = None
        self.process_name = "run.py"
        
    def get_boot_time(self):
        """Get system boot time"""
        try:
            return psutil.boot_time()
        except:
            return None
    
    def check_vm_restart(self):
        """Check if VM was restarted by comparing boot times"""
        current_boot = self.get_boot_time()
        if current_boot and self.last_boot_time:
            if current_boot > self.last_boot_time:
                return True, current_boot
        return False, current_boot
    
    def find_process(self):
        """Find the run.py process"""
        if self.process_pid:
            try:
                proc = psutil.Process(self.process_pid)
                if proc.is_running() and self.process_name in ' '.join(proc.cmdline()):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Search for process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if self.process_name in cmdline and 'python' in cmdline.lower():
                    self.process_pid = proc.info['pid']
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return None
    
    def get_memory_info(self):
        """Get memory usage"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'swap_total_gb': swap.total / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent
        }
    
    def get_cpu_info(self):
        """Get CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None
        return {
            'percent': cpu_percent,
            'count': cpu_count,
            'per_core': cpu_per_core,
            'load_avg': load_avg
        }
    
    def get_disk_info(self):
        """Get disk usage"""
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': disk.percent
        }
    
    def get_process_info(self, proc):
        """Get process-specific info"""
        try:
            mem_info = proc.memory_info()
            cpu_percent = proc.cpu_percent(interval=0.1)
            num_threads = proc.num_threads()
            open_files = len(proc.open_files())
            connections = len(proc.connections())
            
            # Try to get GPU memory if available (for PyTorch)
            gpu_memory = None
            try:
                if platform.system() != 'Windows':
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines:
                            gpu_mem = lines[0].split(',')
                            gpu_memory = {
                                'used_mb': int(gpu_mem[0].strip()),
                                'total_mb': int(gpu_mem[1].strip()),
                                'percent': (int(gpu_mem[0].strip()) / int(gpu_mem[1].strip())) * 100
                            }
                else:
                    # Windows: try nvidia-smi
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
                        capture_output=True, text=True, timeout=5, shell=True
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines:
                            gpu_mem = lines[0].split(',')
                            gpu_memory = {
                                'used_mb': int(gpu_mem[0].strip()),
                                'total_mb': int(gpu_mem[1].strip()),
                                'percent': (int(gpu_mem[0].strip()) / int(gpu_mem[1].strip())) * 100
                            }
            except:
                pass
            
            return {
                'pid': proc.pid,
                'status': proc.status(),
                'memory_rss_gb': mem_info.rss / (1024**3),
                'memory_vms_gb': mem_info.vms / (1024**3),
                'cpu_percent': cpu_percent,
                'num_threads': num_threads,
                'open_files': open_files,
                'connections': connections,
                'gpu_memory': gpu_memory,
                'create_time': datetime.fromtimestamp(proc.create_time()).isoformat()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return {'error': str(e), 'status': 'not_found'}
    
    def check_alerts(self, metrics):
        """Check if any thresholds are exceeded"""
        alerts = []
        
        if metrics['memory']['percent'] > self.alert_thresholds['memory_percent']:
            alerts.append(f"HIGH MEMORY: {metrics['memory']['percent']:.1f}% (threshold: {self.alert_thresholds['memory_percent']}%)")
        
        if metrics['cpu']['percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append(f"HIGH CPU: {metrics['cpu']['percent']:.1f}% (threshold: {self.alert_thresholds['cpu_percent']}%)")
        
        if metrics['disk']['percent'] > self.alert_thresholds['disk_percent']:
            alerts.append(f"HIGH DISK: {metrics['disk']['percent']:.1f}% (threshold: {self.alert_thresholds['disk_percent']}%)")
        
        if metrics['memory']['swap_percent'] > 80:
            alerts.append(f"HIGH SWAP: {metrics['memory']['swap_percent']:.1f}% - System is using swap memory heavily!")
        
        if metrics.get('process') and metrics['process'].get('status') == 'zombie':
            alerts.append("PROCESS ZOMBIE: Process is in zombie state!")
        
        if not metrics.get('process') or metrics['process'].get('status') == 'not_found':
            alerts.append("PROCESS NOT FOUND: run.py process is not running!")
        
        return alerts
    
    def log_metrics(self, metrics, alerts):
        """Log metrics to JSONL file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'metrics': metrics,
            'alerts': alerts
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def print_status(self, metrics, alerts, vm_restarted):
        """Print current status to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"MONITOR STATUS - {timestamp}")
        print(f"{'='*70}")
        
        if vm_restarted:
            print("⚠️  VM RESTART DETECTED!")
        
        print(f"\nSystem Resources:")
        print(f"  Memory: {metrics['memory']['used_gb']:.2f}GB / {metrics['memory']['total_gb']:.2f}GB ({metrics['memory']['percent']:.1f}%)")
        print(f"  Swap: {metrics['memory']['swap_used_gb']:.2f}GB / {metrics['memory']['swap_total_gb']:.2f}GB ({metrics['memory']['swap_percent']:.1f}%)")
        print(f"  CPU: {metrics['cpu']['percent']:.1f}% (cores: {metrics['cpu']['count']})")
        print(f"  Disk: {metrics['disk']['used_gb']:.2f}GB / {metrics['disk']['total_gb']:.2f}GB ({metrics['disk']['percent']:.1f}%)")
        
        if metrics.get('process'):
            proc_info = metrics['process']
            if 'error' not in proc_info:
                print(f"\nProcess (run.py):")
                print(f"  PID: {proc_info['pid']}")
                print(f"  Status: {proc_info['status']}")
                print(f"  Memory: {proc_info['memory_rss_gb']:.2f}GB RSS, {proc_info['memory_vms_gb']:.2f}GB VMS")
                print(f"  CPU: {proc_info['cpu_percent']:.1f}%")
                print(f"  Threads: {proc_info['num_threads']}")
                if proc_info.get('gpu_memory'):
                    gpu = proc_info['gpu_memory']
                    print(f"  GPU Memory: {gpu['used_mb']}MB / {gpu['total_mb']}MB ({gpu['percent']:.1f}%)")
            else:
                print(f"\nProcess (run.py): NOT FOUND")
        else:
            print(f"\nProcess (run.py): NOT FOUND")
        
        if alerts:
            print(f"\n⚠️  ALERTS:")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print(f"\n✓ All systems normal")
        
        print(f"{'='*70}\n")
    
    def run(self):
        """Main monitoring loop"""
        print(f"Starting monitor for {self.process_name}")
        print(f"Log file: {self.log_file.absolute()}")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check for VM restart
                vm_restarted, new_boot_time = self.check_vm_restart()
                if vm_restarted:
                    self.last_boot_time = new_boot_time
                
                # Gather metrics
                metrics = {
                    'memory': self.get_memory_info(),
                    'cpu': self.get_cpu_info(),
                    'disk': self.get_disk_info(),
                    'vm_restarted': vm_restarted
                }
                
                # Find and monitor process
                proc = self.find_process()
                if proc:
                    metrics['process'] = self.get_process_info(proc)
                else:
                    metrics['process'] = {'status': 'not_found'}
                
                # Check alerts
                alerts = self.check_alerts(metrics)
                
                # Log and print
                self.log_metrics(metrics, alerts)
                self.print_status(metrics, alerts, vm_restarted)
                
                # Wait for next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")
        except Exception as e:
            print(f"\n\nMonitor error: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor run.py training process")
    parser.add_argument('--log-file', type=str, default='monitor_log.jsonl',
                       help='Log file path (default: monitor_log.jsonl)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    parser.add_argument('--memory-threshold', type=float, default=90,
                       help='Memory alert threshold percentage (default: 90)')
    parser.add_argument('--cpu-threshold', type=float, default=95,
                       help='CPU alert threshold percentage (default: 95)')
    parser.add_argument('--disk-threshold', type=float, default=90,
                       help='Disk alert threshold percentage (default: 90)')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(
        log_file=args.log_file,
        check_interval=args.interval,
        alert_thresholds={
            'memory_percent': args.memory_threshold,
            'cpu_percent': args.cpu_threshold,
            'disk_percent': args.disk_threshold
        }
    )
    
    monitor.run()


if __name__ == "__main__":
    main()


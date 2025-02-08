"""GPU monitoring and checkpoint utilities for parallel training with comprehensive progress tracking."""
import torch
import psutil
import os
import signal
import threading
import shutil
from pathlib import Path
import time
import json
from typing import Dict, Optional, Tuple
from datetime import datetime
import boto3

class GPUMonitor:
    def __init__(self, gpu_id: int, threshold: float = 0.90):
        self.gpu_id = gpu_id
        self.threshold = threshold
        self.warning_event = threading.Event()
        self._stop_event = threading.Event()
        self._monitor_thread = None

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage."""
        allocated = torch.cuda.memory_allocated(self.gpu_id)
        reserved = torch.cuda.memory_reserved(self.gpu_id)
        total = torch.cuda.get_device_properties(self.gpu_id).total_memory
        return allocated / total, reserved / total

    def monitor_memory(self):
        """Monitor GPU memory usage in a separate thread."""
        while not self._stop_event.is_set():
            allocated_ratio, _ = self.get_gpu_memory_usage()
            if allocated_ratio > self.threshold:
                self.warning_event.set()
            time.sleep(1)  # Check every second

    def start(self):
        """Start GPU monitoring."""
        self._monitor_thread = threading.Thread(target=self.monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def stop(self):
        """Stop GPU monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()

class TrainingProgressMonitor:
    def __init__(self, log_dir: str, model_idx: int):
        self.progress_file = Path(log_dir) / "progress" / f"model_{model_idx}_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()
        self.spot_prices_history = []
        
    def _load_history(self) -> Dict:
        """Load training history including interruptions."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'sessions': [],
            'total_epochs': 0,
            'total_training_time': 0,
            'interruptions': 0,
            'current_instance': None,
            'instance_type_history': {},
            'cost_estimate': 0.0,
            'last_checkpoint': None,
            'spot_price_history': []
        }

    def _get_current_spot_price(self) -> Optional[float]:
        """Get current spot price for the instance."""
        try:
            client = boto3.client('sagemaker')
            instance_id = os.environ.get('SAGEMAKER_INSTANCE_ID')
            if instance_id:
                response = client.describe_notebook_instance(
                    NotebookInstanceName=instance_id
                )
                if 'InstanceMetadata' in response:
                    return response['InstanceMetadata'].get('SpotPrice')
        except Exception as e:
            print(f"Warning: Could not fetch spot price: {e}")
        return None
    
    def start_session(self):
        """Record new training session with spot price tracking."""
        gpu_name = torch.cuda.get_device_name(0)
        instance_type = self._detect_instance_type(gpu_name)
        spot_price = self._get_current_spot_price()
        
        session = {
            'start_time': time.time(),
            'start_datetime': datetime.now().isoformat(),
            'instance_type': instance_type,
            'gpu_type': gpu_name,
            'start_epoch': self.history['total_epochs'],
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
            'checkpoints': [],
            'spot_price': spot_price,
            'spot_price_history': []
        }
        self.history['sessions'].append(session)
        self.history['current_instance'] = instance_type
        
        # Update instance type usage history
        if instance_type not in self.history['instance_type_history']:
            self.history['instance_type_history'][instance_type] = {
                'total_hours': 0,
                'average_spot_price': spot_price if spot_price else 0,
                'price_samples': 1 if spot_price else 0
            }
        
        self._save_history()
        print(f"\nStarting new training session on {instance_type} ({gpu_name})")
        if spot_price:
            print(f"Current spot price: ${spot_price:.3f}/hour")
    
    def _detect_instance_type(self, gpu_name: str) -> str:
        """Detect instance type from GPU name."""
        gpu_to_instance = {
            'T4': 'g4dn',
            'A10G': 'g5',
            'V100': 'p3'
        }
        for gpu, instance in gpu_to_instance.items():
            if gpu in gpu_name:
                return instance
        return 'unknown'
    
    def update_progress(self, epoch: int, metrics: Dict):
        """Update training progress and calculate costs."""
        current_session = self.history['sessions'][-1]
        current_session['current_epoch'] = epoch
        current_session['latest_metrics'] = metrics
        
        # Update total epochs
        self.history['total_epochs'] = epoch
        
        # Calculate and update training time
        current_time = time.time()
        session_duration = (current_time - current_session['start_time']) / 3600  # in hours
        self.history['total_training_time'] = sum(
            (s.get('end_time', current_time) - s['start_time']) / 3600 
            for s in self.history['sessions']
        )
        
        # Update spot price
        spot_price = self._get_current_spot_price()
        if spot_price:
            current_session['spot_price_history'].append({
                'timestamp': datetime.now().isoformat(),
                'price': spot_price
            })
        
        # Update instance type usage and costs
        instance_type = current_session['instance_type']
        inst_history = self.history['instance_type_history'][instance_type]
        inst_history['total_hours'] = \
            self.history['instance_type_history'].get(instance_type, {}).get('total_hours', 0) + session_duration
        
        if spot_price:
            avg_price = inst_history.get('average_spot_price', 0)
            samples = inst_history.get('price_samples', 0)
            new_avg = (avg_price * samples + spot_price) / (samples + 1)
            inst_history['average_spot_price'] = new_avg
            inst_history['price_samples'] = samples + 1
            
            # Update total cost estimate
            self.history['cost_estimate'] = sum(
                inst['total_hours'] * inst['average_spot_price']
                for inst in self.history['instance_type_history'].values()
            )
        
        self._save_history()
    
    def record_checkpoint(self, checkpoint_path: str, epoch: int, batch_idx: int):
        """Record checkpoint information."""
        current_session = self.history['sessions'][-1]
        checkpoint_info = {
            'path': checkpoint_path,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'time': time.time(),
            'spot_price': self._get_current_spot_price()
        }
        current_session['checkpoints'].append(checkpoint_info)
        self.history['last_checkpoint'] = checkpoint_info
        self._save_history()
    
    def record_interruption(self):
        """Record spot interruption."""
        current_session = self.history['sessions'][-1]
        current_time = time.time()
        current_session['end_time'] = current_time
        current_session['end_datetime'] = datetime.now().isoformat()
        current_session['interrupted'] = True
        current_session['spot_price_at_interruption'] = self._get_current_spot_price()
        self.history['interruptions'] += 1
        self._save_history()
        print(f"\nSpot interruption recorded for {current_session['instance_type']}")
    
    def _save_history(self):
        """Save training history to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def print_summary(self):
        """Print comprehensive training summary."""
        print("\n=== Training History Summary ===")
        print(f"Total Epochs Completed: {self.history['total_epochs']}")
        print(f"Total Training Time: {self.history['total_training_time']:.2f} hours")
        print(f"Number of Interruptions: {self.history['interruptions']}")
        print(f"Estimated Spot Cost: ${self.history['cost_estimate']:.2f}")
        
        print("\nInstance Type Usage:")
        for instance_type, info in self.history['instance_type_history'].items():
            print(f"\n{instance_type}:")
            print(f"  Hours: {info['total_hours']:.2f}")
            print(f"  Average Spot Price: ${info['average_spot_price']:.3f}/hour")
            print(f"  Total Cost: ${info['total_hours'] * info['average_spot_price']:.2f}")
        
        print("\nTraining Sessions:")
        for i, session in enumerate(self.history['sessions']):
            duration = (session.get('end_time', time.time()) - session['start_time']) / 3600
            print(f"\nSession {i+1}:")
            print(f"Instance Type: {session['instance_type']}")
            print(f"GPU Type: {session['gpu_type']}")
            print(f"Duration: {duration:.2f} hours")
            print(f"Starting Spot Price: ${session.get('spot_price', 0):.3f}/hour")
            print(f"Epochs: {session['start_epoch']} -> {session.get('current_epoch', session['start_epoch'])}")
            print(f"Checkpoints: {len(session['checkpoints'])}")
            if session.get('interrupted', False):
                print(f"Status: Interrupted (Spot price: ${session.get('spot_price_at_interruption', 0):.3f}/hour)")
            print(f"Latest Metrics: {session.get('latest_metrics', {})}")

class CheckpointManager:
    def __init__(self, log_dir: str, model_idx: int, progress_monitor: Optional[TrainingProgressMonitor] = None):
        self.checkpoint_dir = Path(log_dir) / "checkpoints" / str(model_idx)
        self.emergency_dir = Path(log_dir) / "emergency_checkpoints" / str(model_idx)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.emergency_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint = None
        self.progress_monitor = progress_monitor
        self.interruption_count = 0
        self._setup_interruption_handler()

    def _setup_interruption_handler(self):
        """Enhanced interruption handler with emergency checkpointing."""
        def handler(signum, frame):
            self.interruption_count += 1
            print(f"\nSpot interruption #{self.interruption_count} detected at {datetime.now().isoformat()}")
            
            if self.latest_checkpoint:
                # Create emergency checkpoint
                emergency_path = self.emergency_dir / f"emergency_checkpoint_{int(time.time())}.pt"
                try:
                    # Copy latest checkpoint to emergency location
                    shutil.copy2(self.latest_checkpoint, emergency_path)
                    print(f"Emergency checkpoint saved: {emergency_path}")
                except Exception as e:
                    print(f"Error saving emergency checkpoint: {e}")
                
                if self.progress_monitor:
                    self.progress_monitor.record_interruption()
                    
            # Save interruption metadata
            interruption_meta = {
                'timestamp': datetime.now().isoformat(),
                'latest_checkpoint': str(self.latest_checkpoint) if self.latest_checkpoint else None,
                'emergency_checkpoint': str(emergency_path) if 'emergency_path' in locals() else None,
                'interruption_number': self.interruption_count
            }
            meta_path = self.emergency_dir / 'interruption_metadata.json'
            with open(meta_path, 'a') as f:
                json.dump(interruption_meta, f)
                f.write('\n')
                
        # Register handlers for both SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def save_checkpoint(
        self, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        batch_idx: int,
        train_config: Dict,
        metrics: Dict
    ) -> str:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_e{epoch}_b{batch_idx}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_config': train_config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.latest_checkpoint = checkpoint_path
        
        if self.progress_monitor:
            self.progress_monitor.record_checkpoint(str(checkpoint_path), epoch, batch_idx)
        
        return str(checkpoint_path)

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Enhanced checkpoint loading with emergency recovery."""
        # First check emergency checkpoints
        emergency_checkpoints = sorted(self.emergency_dir.glob('emergency_checkpoint_*.pt'))
        if emergency_checkpoints:
            latest_emergency = emergency_checkpoints[-1]
            print(f"\nFound emergency checkpoint: {latest_emergency}")
            try:
                checkpoint = torch.load(latest_emergency)
                self.latest_checkpoint = latest_emergency
                return checkpoint
            except Exception as e:
                print(f"Error loading emergency checkpoint: {e}")
        
        # Fall back to regular checkpoints
        if not self.latest_checkpoint or not self.latest_checkpoint.exists():
            # Try to find the latest checkpoint in the directory
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
            if not checkpoints:
                return None
            self.latest_checkpoint = checkpoints[-1]
            
        try:
            return torch.load(self.latest_checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
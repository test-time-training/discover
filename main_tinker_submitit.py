import os
import sys
import subprocess
import signal
import time
from datetime import datetime
import socket
from pathlib import Path
import submitit
import hydra
import argparse
import shutil
from typing import Optional
os.environ["SBATCH_REQUEUE"] = "1"


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        # Handle strings like "gpu:8" or "gpu:4,vmem:..."
        digits = "".join(ch for ch in v if ch.isdigit())
        return int(digits) if digits else None

def _detect_expected_total_gpus() -> int:
    """
    Decide how many GPUs we *expect* across the whole job.
    Priority:
      1) EXPECT_TOTAL_GPUS (override)
      2) SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE
      3) SLURM_GPUS (some setups expose total GPUs for the job)
      4) CUDA_VISIBLE_DEVICES count (single-node fallback)
      5) default 1 (so we don't block if nothing is set)
    """
    # Explicit override
    override = _env_int("EXPECT_TOTAL_GPUS")
    if override is not None and override > 0:
        return override

    num_nodes = _env_int("SLURM_JOB_NUM_NODES") or 1
    gpn = _env_int("SLURM_GPUS_ON_NODE")
    if gpn and num_nodes:
        return num_nodes * gpn

    slurm_gpus_total = _env_int("SLURM_GPUS")
    if slurm_gpus_total:
        return slurm_gpus_total

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [x for x in cvd.split(",") if x.strip() and x.strip().lower() != "none"]
        if ids:
            return len(ids)

    return 1


def wait_for_ray_cluster():
    """
    Connects to the Ray cluster and waits until the cluster has the expected number
    of GPUs registered. Set EXPECT_TOTAL_GPUS to override detection if your Slurm
    layout is heterogeneous.
    """
    import ray # try to only import ray once

    timeout_s: int = 300
    poll_s: float = 5.0
    expected = _detect_expected_total_gpus()

    # If we only expect 1 node, don't spin unnecessarily.
    if expected <= 8:
        return

    addr = os.environ.get("RAY_ADDRESS", "auto")
    ns = os.environ.get("SLURM_JOB_ID") or "default"
    print(f"Connecting to Ray at {addr!r} (namespace={ns})...")
    ray.init(address=addr, namespace=str(ns), ignore_reinit_error=True, logging_level="ERROR")

    def _ray_total_registered_gpus() -> int:
        """
        Query Ray for the total number of GPUs registered on alive nodes.
        Tries multiple Ray APIs for compatibility across Ray versions.
        """
        try:
            # Ray <= 2.x: ray.nodes() returns list of dicts with "Resources" and "Alive".
            nodes = ray.nodes()
            return int(sum(n.get("Resources", {}).get("GPU", 0) for n in nodes if n.get("Alive", False)))
        except Exception:
            pass

        try:
            # Fallback: cluster_resources is a global summary
            res = ray.cluster_resources()
            return int(res.get("GPU", 0))
        except Exception:
            return 0

    print(f"Waiting for Ray to register GPUs: expected >= {expected}")
    waited = 0.0
    last = -1
    while waited < timeout_s:
        total_gpus = _ray_total_registered_gpus()
        if total_gpus != last:
            print(f"  progress: {total_gpus}/{expected} GPUs registered")
            last = total_gpus

        if total_gpus >= expected:
            print(f"Ray cluster ready with {total_gpus} GPUs (expected {expected}).")
            return

        time.sleep(poll_s)
        waited += poll_s

    raise RuntimeError(
        f"Timed out after {timeout_s}s waiting for Ray GPUs: "
        f"registered {last}, expected >= {expected}. "
        f"Tip: set EXPECT_TOTAL_GPUS to override detection if your nodes differ."
    )


def setup_ray_cluster():
    """Setup Ray cluster exactly like the sbatch script."""
    # Get SLURM environment variables
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    node_rank = int(os.environ.get('SLURM_PROCID', 0))
    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    
    if num_nodes == 1:
        # Single node - let main_grpo.py handle Ray init
        return
    
    # Multi-node setup following the sbatch script exactly
    
    # Get nodes array (equivalent to: mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    result = subprocess.run(['scontrol', 'show', 'hostnames', node_list], capture_output=True, text=True, check=True)
    nodes_array = result.stdout.strip().split('\n')
    head_node = nodes_array[0]
    
    print(f"Number of workers: {num_nodes}")
    
    # Get IP address (equivalent to: ip=$(getent ahostsv4 "$head_node" | awk 'NR==1{print $1}'))
    result = subprocess.run(['getent', 'ahostsv4', head_node], capture_output=True, text=True, check=True)
    ip = result.stdout.strip().split()[0]
    
    # Use exact same ports as sbatch script
    port = 6379
    dashboard_port = 8265
    client_port = 10001
    ip_head = f"{ip}:{port}"
    
    print(f"Head node: {head_node}  IP: {ip}  GCS: {ip_head}")
    
    current_hostname = socket.gethostname()

    job_id = os.environ.get('SLURM_JOB_ID')  # unique id shared by nodes under the same job
    ray_status_dir = Path(f"./submitit_jobs/ray_tmp/{job_id}")
    ray_status_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    # Expose client address for ray.init("auto") discovery
    os.environ.setdefault("RAY_ADDRESS", f"ray://{ip}:{client_port}")
    
    # Reserve explicit, non-overlapping ports to avoid collisions
    worker_port_min = int(os.environ.get("RAY_MIN_WORKER_PORT", "20000"))
    worker_port_max = int(os.environ.get("RAY_MAX_WORKER_PORT", "25999"))
    node_manager_port = int(os.environ.get("RAY_NODE_MANAGER_PORT", "62365"))
    object_manager_port = int(os.environ.get("RAY_OBJECT_MANAGER_PORT", "62366"))
    metrics_export_port = int(os.environ.get("RAY_METRICS_EXPORT_PORT", "62367"))
    runtime_env_agent_port = int(os.environ.get("RAY_RUNTIME_ENV_AGENT_PORT", "62368"))
    dashboard_agent_grpc_port = int(os.environ.get("RAY_DASHBOARD_AGENT_GRPC_PORT", "62470"))
    
    if node_rank == 0:
        # Start Ray head (equivalent to the srun ray start --head command)
        print(f"STARTING HEAD at {head_node}")
        # Ensure a clean state in case of requeues or leftovers
        try:
            subprocess.run(['ray', 'stop', '--force'], check=False)
        except Exception:
            pass
        cmd = [
            'ray', 'start', '--head',
            f'--node-ip-address={ip}',
            f'--port={port}',
            '--dashboard-host=127.0.0.1',
            f'--dashboard-port={dashboard_port}',
            f'--ray-client-server-port={client_port}',
            f'--node-manager-port={node_manager_port}',
            f'--object-manager-port={object_manager_port}',
            f'--metrics-export-port={metrics_export_port}',
            f'--runtime-env-agent-port={runtime_env_agent_port}',
            f'--dashboard-agent-grpc-port={dashboard_agent_grpc_port}',
            f'--min-worker-port={worker_port_min}',
            f'--max-worker-port={worker_port_max}',
            '--block'
        ]
        subprocess.Popen(cmd)
        time.sleep(30)
        
        # Create a marker file for coordination
        ray_file_path = f'{ray_status_dir}/ray_ready_rank_{node_rank}'
        with open(ray_file_path, 'w') as f:
            f.write(ip_head)
        os.chmod(ray_file_path, 0o777)
        print(f"HEAD at {head_node} HAS STARTED")
            
    else:
        # Worker node - wait for head and then start worker
        print(f"STARTING WORKER {node_rank} at {current_hostname}")
        # Wait for head node to be ready
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            if os.path.exists(f'{ray_status_dir}/ray_ready_rank_0'):
                break
            print(f"WORKER {node_rank} at {current_hostname} STILL WAITING FOR HEAD")
            time.sleep(20)
            wait_time += 1
        
        if wait_time >= max_wait:
            raise RuntimeError("Timeout waiting for Ray head node")
        
        # Ensure a clean state in case of requeues or leftovers
        try:
            subprocess.run(['ray', 'stop', '--force'], check=False)
        except Exception:
            pass
        cmd = [
            'ray', 'start',
            f'--address={ip_head}',
            f'--node-manager-port={node_manager_port}',
            f'--object-manager-port={object_manager_port}',
            f'--metrics-export-port={metrics_export_port}',
            f'--runtime-env-agent-port={runtime_env_agent_port}',
            f'--dashboard-agent-grpc-port={dashboard_agent_grpc_port}',
            f'--min-worker-port={worker_port_min}',
            f'--max-worker-port={worker_port_max}',
            '--block'
        ]
        subprocess.Popen(cmd)
        time.sleep(30)

        ray_file_path = f'{ray_status_dir}/ray_ready_rank_{node_rank}'
        with open(ray_file_path, 'w') as f:
            f.write(ip_head)
        os.chmod(ray_file_path, 0o777)
        print(f"WORKER {node_rank} at {current_hostname} HAS STARTED")


def setup_environment():
    project_root = os.getcwd()
    os.environ["PYTHONPATH"] = (
        project_root
        + ":" 
        + os.environ.get("PYTHONPATH", "")
    )
    print("PYTHONPATH:", os.environ["PYTHONPATH"])
    """Setup necessary environment variables for distributed training."""
    # Set up Ray and other environment variables as needed
    os.environ.setdefault("RAY_raylet_start_wait_time_s", "300") # Allow buffer for startup time
    os.environ.setdefault("RAY_DEDUP_LOGS", "1") # Should dedup logs when possible
    os.environ.setdefault("RAY_COLOR_PREFIX", "0") # Do not color, it looks weird in logs
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1") # Fix for torch dynamo crash
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    os.environ.setdefault("VLLM_LOGGING_CONFIG_PATH", "./config/logging/vllm.json") # prevent vllm spam
    
    # Cluster requirements
    os.environ["NCCL_IB_TIMEOUT"] = "20"
    os.environ["NCCL_IB_SL"] = "0"
    os.environ["NCCL_IB_TC"] = "41"

    # CRITICAL: Set up GPU visibility for Ray
    # SLURM allocates GPUs but may not set CUDA_VISIBLE_DEVICES properly
    if 'SLURM_GPUS_PER_NODE' in os.environ:
        print('SLURM_GPUS_PER_NODE In OS.ENVIRONMENT')
        gpus_per_node = int(os.environ['SLURM_GPUS_PER_NODE'])
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            # Set CUDA_VISIBLE_DEVICES to all allocated GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(gpus_per_node)))
            print(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            print(os.environ['CUDA_VISIBLE_DEVICES'])
    
    # Alternative: If SLURM_GPUS_ON_NODE is available, use that
    elif 'SLURM_GPUS_ON_NODE' in os.environ:
        print('SLURM_GPUS_ON_NODE In OS.ENVIRONMENT')
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_GPUS_ON_NODE']
        print(f"Set CUDA_VISIBLE_DEVICES from SLURM_GPUS_ON_NODE: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Setup Ray cluster coordination for multi-node (following sbatch script)
    setup_ray_cluster()


class TrainingJob(submitit.helpers.Checkpointable):
    """Submitit job class for running tinker_cookbook.recipes.ttt.train."""
    
    def __init__(self, hydra_args, working_dir, max_requeue=4, requeue_count=0):
        self.hydra_args = hydra_args
        self.working_dir = working_dir
        self.max_requeue = max_requeue
        self.requeue_count = requeue_count
        self._cancelled_by_user = False
        signal.signal(signal.SIGTERM, self._on_sigterm)  # scancel default
    
    def _on_sigterm(self, signum, frame):
        # Mark as user cancellation and exit fast; submitit may still call checkpoint()
        self._cancelled_by_user = True
        raise SystemExit(0)
    
    def _cleanup_before_restart(self):
        job_id = os.environ.get("SLURM_JOB_ID")
        node_rank = int(os.environ.get("SLURM_PROCID", 0))
        if node_rank == 0 and job_id:
            ray_status_dir = Path(f"./submitit_jobs/ray_tmp/{job_id}")
            shutil.rmtree(ray_status_dir, ignore_errors=True)

    def _should_requeue(self) -> bool:
        return max(self.requeue_count, int(os.environ.get("SLURM_RESTART_COUNT", 0))) < self.max_requeue

    def _request_requeue_now(self):
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return False
        try:
            # This transitions the current RUNNING job back to PENDING and Slurm will kill us.
            subprocess.run(["scontrol", "requeue", job_id], check=True)
            return True
        except Exception as e:
            print(f"[submitit] scontrol requeue failed: {e}", flush=True)
            return False
    
    def __call__(self):
        """Main execution function called by submitit."""

        # Change to working directory
        os.chdir(self.working_dir)
        
        # Setup environment and Ray cluster (following sbatch script)
        setup_environment()

        # Sync to make sure ray is ready on all nodes
        wait_for_ray_cluster()
        
        # Import and run the main function
        sys.path.insert(0, str(self.working_dir))
        
        # Add nnodes parameter like in sbatch script
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        node_rank = int(os.environ.get('SLURM_PROCID', 0))
        try:
            
            if node_rank == 0:
                # Build command from hydra args passed on the CLI
                cmd = [
                    sys.executable, "-m", "tinker_cookbook.recipes.ttt.train",
                    *self.hydra_args,
                ]
                print("Running training command:")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)
            
            else:
                # Worker nodes - stay alive for Ray coordination
                print(f"Worker node {node_rank}: Staying alive for Ray workers...")
                while True:
                    time.sleep(60)  # Keep process alive to participate in Ray cluster
        
        except BaseException as e:
            print("=== EXCEPTION CAUGHT: auto-requeue path ===", flush=True)
            import traceback
            traceback.print_exc()
            
            self._cleanup_before_restart()

            if isinstance(e, (SystemExit, KeyboardInterrupt)) or self._cancelled_by_user:
                print("User cancelled via SIGTERM. Not requeuing.", flush=True)
                raise

            if self._should_requeue():
                ok = self._request_requeue_now()
                if ok:
                    while True:
                        time.sleep(60)
                else:
                    raise
            else:
                print(f"Max requeues ({self.max_requeue}) reached. Not requeuing.", flush=True)
                raise
    
    def checkpoint(self):
        print("=== CHECKPOINT CALLED ===")
        job_id = os.environ.get('SLURM_JOB_ID')  # unique id shared by nodes under the same job
        node_rank = int(os.environ.get('SLURM_PROCID', 0))
        if node_rank == 0:
            ray_status_dir = Path(f"./submitit_jobs/ray_tmp/{job_id}")
            # remove ray sync files since the requeued job has the same job id and need to re-setup Ray
            shutil.rmtree(ray_status_dir, ignore_errors=True)

        if self.requeue_count >= self.max_requeue:
            print(f"Max requeues ({self.max_requeue}) reached. Stopping.")
            return None

        next_job = TrainingJob(
            hydra_args=self.hydra_args,
            working_dir=self.working_dir,
            max_requeue=self.max_requeue,
            requeue_count=self.requeue_count + 1,
        )
        return submitit.helpers.DelayedSubmission(next_job)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch tinker_cookbook.recipes.ttt.train with submitit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Submitit/SLURM specific arguments
    parser.add_argument("--job-dir", type=str, default="./submitit_jobs",  help="Directory to store submitit job files")  # please don't change
    parser.add_argument("--job-name", type=str, default="submitit-test", help="SLURM job name")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes to request (default: 2 like sbatch)")
    parser.add_argument("--gpus-per-node", type=int, default=0, help="Number of GPUs per node (default: 0 like sbatch)")
    parser.add_argument("--cpus-per-task", type=int, default=128, help="Number of CPUs per task (default: 128 like sbatch)")
    parser.add_argument("--mem", type=str, default="0", help="Memory per node (e.g., '64GB', '0' for all available)")
    parser.add_argument("--timeout_min", type=int, default=240, help="Time limit (unit: min)")
    parser.add_argument("--partition", type=str, default="default", help="SLURM partition")
    parser.add_argument("--account", type=str, default="default", help="SLURM account")
    parser.add_argument("--no-exclusive", action="store_true", help="Don't request exclusive usage of resources")
    
    # Auto-requeue options
    parser.add_argument("--max-requeue", type=int, default=10, help="Maximum number of requeues")
    
    # Other options
    parser.add_argument("--dry-run", action="store_true", help="Print job configuration without submitting")
    parser.add_argument("--local", action="store_true", help="Run locally instead of submitting to SLURM")
    
    # Parse known args to separate submitit args from hydra overrides
    args, unknown_args = parser.parse_known_args()
    
    return args, unknown_args


def submit_job(args, hydra_overrides):
    """Submit the job using submitit."""

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    # Setup job directory
    job_dir = Path(args.job_dir) / date_str / time_str
    if not args.local:
        job_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=job_dir, cluster="slurm")
    
    # Configure SLURM parameters
    slurm_kwargs = {
        "slurm_job_name": args.job_name,
        "timeout_min": args.timeout_min,
        "slurm_partition": args.partition,
        "slurm_account": args.account,
        "nodes": args.nodes,
        "tasks_per_node": 1,
    }
    hostname = socket.gethostname()
    if 'cluster_a' not in hostname and args.partition != "owners" and args.partition != "preempt":
        slurm_kwargs["gpus_per_node"] = args.gpus_per_node

    slurm_kwargs["cpus_per_task"] = args.cpus_per_task
    slurm_kwargs["slurm_mem"] = "0"
    
    if args.mem != "0":
        slurm_kwargs["mem"] = args.mem
    
    # Add optional SLURM parameters
    slurm_additional = {
        "output": f"{job_dir}/%j.out",
        "error": f"{job_dir}/%j.err",
    }
    if not args.no_exclusive:
        slurm_additional["exclusive"] = ""

    # Remove None values
    slurm_kwargs = {k: v for k, v in slurm_kwargs.items() if v is not None}
    
    # Configure executor
    executor.update_parameters(
        slurm_additional_parameters=slurm_additional,
        **slurm_kwargs
    )
    
    executor.update_parameters(
        slurm_signal_delay_s=120,  # Send signal-2 2 minutes before timeout
    )
    
    # Create job instance
    job = TrainingJob(hydra_overrides, working_dir=os.getcwd(), max_requeue=args.max_requeue)
    
    if args.dry_run:
        print("Job configuration:")
        print(f"  Job directory: {job_dir}")
        print(f"  SLURM parameters: {slurm_kwargs}")
        print(f"  Hydra overrides: {hydra_overrides}")
        print(f"  Auto-requeue: {args.auto_requeue} (max: {args.max_requeue})")
        return None
    
    if args.local:
        print("Running locally...")
        job()
        return None
    
    # Submit job
    submitted_job = executor.submit(job)
    
    return submitted_job


def main():
    """Main function."""
    args, hydra_overrides = parse_arguments()
    
    print(f"Hydra overrides: {hydra_overrides}")
    
    # Submit the job
    submit_job(args, hydra_overrides)


if __name__ == "__main__":
    main()
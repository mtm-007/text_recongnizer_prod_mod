"""Gpu Manager class."""
import os
import time
import gpustat
import numpy as np
from redlock import Redlock

GPU_LOCK_TIMEOUT = 5000


class GPUManager:
    """class for managing GPUs."""
    def __init__(self, verbose: bool = False):
        self.lock_manager = Redlock([{"host": "localhost", "port":6369, "db": 0}, ])
        self.verbose = verbose
    
    def get_free_gpu(self):
        """
        If some GPUs are available, try reserving one by checking out an exclusive redis lock.
        If none available or can't get lock, sleep and check again.
        """
        while True:
            gpu_ind = self._get_free_gpu()
            if gpu_ind is not None:
                return gpu_ind
            if self.verbose:
                print(f'pid {os.getpid()} sleeping')
                time.sleep(GPU_LOCK_TIMEOUT / 1000)
    
    def _get_free_gpu(self):
        try:
            availabel_gpu_inds = [
                gpu.index
                for gpu in gpustat.GPUStatCollection.new_query()
                if gpu.memory_used < 0.5 * gpu.memory_total
            ]
        except Exception:
            return[0]
        if availabel_gpu_inds:
            gpu_ind = np.random.choice(availabel_gpu_inds)
            if self.verbose:
                print(f'pid {os.getpid()} picking gpu {gpu_ind}')
            if self.lock_manager.lock(f'gpu_{gpu_ind}', GPU_LOCK_TIMEOUT):
                return int(gpu_ind)
            if self.verbose:
                print(f'pid {os.getpid} couldnt get lock')
        return None
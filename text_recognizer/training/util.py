"""Function to train a model."""
from time import time
from typing import Optional

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from text_recognizer.datasets.dataset import Dataset
from text_recognizer.models.base import Model
from text_recognizer.training.gpu_util_sampler import GPUUtilizationSampler


Early_Stopping = True
GPU_UTIL_SAMPLER = True

def train_model(
        model: Model,
        dataset: Dataset,
        epochs: int,
        batch_size: int,
        gpu_ind: Optional[int] = None,
        use_wandb: bool= False )-> Model:
    """Train Model."""
    callbacks = []

    if Early_Stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilization =GPUUtilizationSampler(gpu_ind)
        callbacks.append(gpu_utilization)

    model.network.summary()


    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print('Training hook {:2f} s'.format(time() - t))

    if GPU_UTIL_SAMPLER and gpu_ind is not None:
        gpu_utilization = gpu_utilization.samples
        print(f'GPU utilizatioin: {round(np.mean(gpu_utilization), 2)} +-{round(np.std(gpu_utilization), 2)}')

    return model 
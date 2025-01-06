from typing import Dict, List

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def window_smooth(data: List[float], window_size: int = 10) -> List[float]:
    """Copy from https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py

    smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k]), where the param 'window_size' equals to 2*k + 1
    """
    if window_size > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(window_size)
        x = np.asarray(data)
        z = np.ones(len(x))
        smooth_data = np.convolve(x, y, "same") / np.convolve(z, y, "same")
        smooth_data = smooth_data.tolist()
    else:
        smooth_data = data
    return smooth_data


def average_smooth(data: List[float], lambda_: float = 0.6) -> List[float]:
    """y[t] = lambda_ * y[t-1] + (1-lambda_) * y[t]"""
    smooth_data = []
    for i in range(len(data)):
        if i == 0:
            smooth_data.append(data[i])
        else:
            smooth_data.append(smooth_data[-1] * lambda_ + data[i] * (1 - lambda_))
    return smooth_data


def tb2dict(tb_file_path: str, keys: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Convert tensorboard log file into a dict of data points"""
    ea = event_accumulator.EventAccumulator(tb_file_path)
    ea.Reload()
    statistics = dict()
    for key in keys:
        assert key in ea.scalars.Keys(), f"{key} is not recorded by the tensorboard!"
        items = ea.scalars.Items(key)
        steps, values = list(), list()
        for item in items:
            steps.append(item.step)
            values.append(item.value)
        statistics[key] = {"steps": steps, "values": values}
    return statistics

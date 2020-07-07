#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:53:57 2018
@author: matthewszhang
"""

import importlib
import importlib.util
import re
import os
import os.path as osp

import metaworld
import logging
import numpy as np


def mt_io_information(n_subtasks):
    if n_subtasks <= 10:
        env = metaworld.benchmarks.ML10.get_train_tasks()
        task = env.sample_tasks(1)
        env.set_task(task[0])
        return np.prod(env.observation_space.shape),\
               np.prod(env.action_space.shape), \
            'continuous'

    else:
        env = metaworld.benchmarks.ML50.get_train_tasks()
        task = env.sample_tasks(1)
        env.set_task(task[0])
        return np.prod(env.observation_space.shape),\
               np.prod(env.action_space.shape), \
            'continuous'


def make_mt_env(task_name, num_subtasks):
    if num_subtasks <= 10:
        env = metaworld.benchmarks.ML10.get_train_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    else:
        env = metaworld.benchmarks.ML50.get_train_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    return env


def make_mt_test(task_name, num_subtasks):
    if num_subtasks <= 10:
        env = metaworld.benchmarks.ML10.get_test_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    else:
        env = metaworld.benchmarks.ML50.get_test_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    return env




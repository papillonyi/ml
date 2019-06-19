# -*- coding: utf-8 -*-
import os
import sagemaker
from sagemaker import get_execution_role

role = 'arn:aws:iam::478824966940:role/service-role/AmazonSageMaker-ExecutionRole-20190516T114455'

from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow(entry_point='cifar10base.py',
                        # script_mode=True,
                       role=role,
                       framework_version='1.12.0',
                       py_version='py3',
                       hyperparameters={'learning_rate': 1e-4, 'decay': 1e-6},
                       # training_steps=1000, evaluation_steps=100,
                       train_instance_count=1, train_instance_type='ml.c4.xlarge')

inputs = "s3://ml-papillonyi"
estimator.fit(inputs)
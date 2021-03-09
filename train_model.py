"""Launch SageMaker training job from local."""
from datetime import datetime
from sagemaker.pytorch.estimator import PyTorch
import os
import boto3
import pandas as pd
import sagemaker

# Define AWS sessions.
sagemaker_session = sagemaker.Session()
# Define role arn with SageMaker and S3 access.
role = f"arn:aws:iam::{os.environ['AWS_ACCOUNT_NUMBER']}:role/SageMakerFullAccess"
# Define S3 variables for data and model storage.
bucket = "brent-sage-dev"
model_prefix = "sagemaker/amazon_review_classifier/train"
input_path = f"s3://{bucket}/{model_prefix}/input_data"
output_path = f"s3://{bucket}/{model_prefix}/model"
code_path = f"s3://{bucket}/{model_prefix}/src"
# Upload data to S3.
sagemaker_session.upload_data(
    path="./data/small_book_reviews.json",
    bucket=bucket,
    key_prefix=f"{model_prefix}/input_data"
)
# Define hyperparameters.
hyperparameters = {
    "input_path": input_path,
    "model_name": "distilbert-base-uncased",
    "train_batch_size": 32,
    "valid_batch_size": 128,
    "epochs": 2,
    "learning_rate": 5e-5,
    "weight_decay": .01,
    "warmup_steps": 500,
    "max_sequence_length": 128
}
# Create SageMaker estimator and laucn training job.
pytorch_estimator = PyTorch(
    entry_point='model.py',
    source_dir='/Users/brent/projects/sagemaker_train_demo/src',
    instance_type='ml.p2.xlarge',
    instance_count=1,
    framework_version='1.5.0',
    py_version='py3',
    hyperparameters=hyperparameters,
    code_location=code_path,
    output_path=output_path,
    role=role
)
pytorch_estimator.fit(inputs=None, job_name=f"amazon-review-model-{datetime.now().strftime('%Y%m%d%H%M%S')}")

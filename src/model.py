"""
This tutorial borrows from https://huggingface.co/transformers/custom_datasets.html#seq-imdb for
"""
import argparse
import os
import sys
import logging

import boto3
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class AmazonReviewDataset(Dataset):
    """Inherits from torch.utils.data.Dataset.

    Arguments:
        data (pd.DataFrame): Contains columns 'input' and 'label'.
        model_name (str): Name of the transformers model to load the tokenizer.
        max_sequence_length (int): Maximum length of the sequence.
    """
    def __init__(self, data, model_name, max_sequence_length):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index):
        """Returns the model inputs at the specified index."""
        text_input = self.data["input"].iloc[index]
        input_dict = self.tokenizer.encode_plus(
            text_input,
            max_length=args.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": input_dict["input_ids"][0],
            "attention_mask": input_dict["attention_mask"][0],
            "labels": torch.tensor(self.data["label"].iloc[index], dtype=torch.long)
        }
        return inputs

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)


def preprocess_data(df):
    """Pre-process data for training."""
    # Convert 5 star scale to binary -- positive (1) or negative (0).
    star_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    df["label"] = df["star_rating"].map(star_map)
    # Concatenate the review headline with the body for model input.
    df["input"] = df["review_headline"] + " " + df["review_body"]
    # Drop nans.
    df = df.loc[df["input"].notnull()].copy()
    # Split data into train and valid.
    traindf, validdf = train_test_split(df[["input", "label"]])
    return traindf, validdf


def compute_metrics(output):
    """Compute custom metrics to track in logging."""
    y_true = output.label_ids
    y_pred = output.predictions.argmax(-1)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    return {
        "precision": precision,
        "recall": recall,
        "auprc": auprc
    }
    
def main(args):
    """Executes training job."""
    # Load data into a pandas DataFrame.
    df = read_object(args.input_path, file_type="json").sample(frac=1.)
    if args.max_data_rows:
        df = df.iloc[:args.max_data_rows].copy()
        
    print(f"Data contains {len(df)} rows")
    # Preprocess and featurize data.
    traindf, validdf = preprocess_data(df)
    # Create data set for model input.
    train_dataset = AmazonReviewDataset(traindf, args.model_name, args.max_sequence_length)
    valid_dataset = AmazonReviewDataset(validdf, args.model_name, args.max_sequence_length)

    # Define model and trainer.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        torchscript=True
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "results"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=500,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.model_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="steps",
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )
    # Train the model.
    trainer.train()
    trainer.evaluate()
    
    # Save the model to the output_path defined in train_model.py.
    save_model(model, args.model_dir)
    
def read_object(s3_path, file_type, **kwargs):
    """Read a file from S3 into a pandas.DataFrame.

    Arguments:
        s3_path (str)

    Returns:
        df (pd.DataFrame)
    """
    # Get S3 client.
    s3 = boto3.client('s3')
    # Get object.
    bucket, key = s3_path.split('/')[2], '/'.join(s3_path.split('/')[3:])
    object = s3.get_object(Bucket=bucket, Key=key)
    # Read object into pandas.DataFrame.
    if file_type == "csv":
        df = pd.read_csv(object['Body'], **kwargs)
    elif file_type == "json":
        df = pd.read_json(object['Body'], lines=True, **kwargs)
    else:
        raise Exception("Only file_type csv and json currently supported.")
    return df


def write_object(df, s3_path, file_type="json"):
    """Save a pandas.DataFrame to S3.

    Arguments:
        df (pd.DataFrame)
        s3_path (str)
    """
    if file_type == "csv":
        df.to_csv(s3_path)
    elif file_type == "json":
        df.to_json(s3_path, lines=True, orient="records")
    else:
        raise Exception("Only file_type csv and json currently supported.")

        
def save_model(model, model_dir):
    logger.info(f"Save model...")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dummy_row = encode_sequences(["dummy"])
    dummy_input = (
        dummy_row["input_ids"][0].unsqueeze(0).to(device),
        dummy_row["attention_mask"][0].unsqueeze(0).to(device)
    )
    traced_model = torch.jit.trace(model.to(device), dummy_input)
    
    model_path = os.path.join(model_dir, "model.pth")
    logger.info(f"Model path: {model_path}")
    
    torch.jit.save(traced_model, model_path)
    
    
tokenizer = None

def get_tokenizer(name):
    global tokenizer
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(name)
        
    return tokenizer

def encode_sequences(sequences):
    tokenizer = get_tokenizer("distilbert-base-uncased")

    return tokenizer.batch_encode_plus(
        sequences,
        max_length=300,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters from launch_training_job.py get passed in as command line args.
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--train_size', type=float, default=.85)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_data_rows', type=int, default=None)
    parser.add_argument('--max_sequence_length', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=128)
    # SageMaker environment variables.
    parser.add_argument('--hosts', type=list, default=os.environ['SM_HOSTS'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR']) # output_path arg from train_model.py.
    parser.add_argument('--num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    # Parse command-line args and run main.
    args = parser.parse_args()
    main(args)

##################################
# SageMaker PyTorch model server

VERSIONS_USE_NEW_API = ["1.5.1"]

    
def model_fn(model_dir):
    logger.info(f"model_fn")
    try:
        model_path = os.path.join(model_dir, "model.pth")
        
        return torch.jit.load(model_path, map_location=torch.device("cpu"))
    
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
        return None
      
        
def input_fn(request_body, request_content_type):
    logger.info(f"input_fn")
    logger.info(f"request_content_type: {request_content_type}")
    logger.info(request_body)
    
    if request_content_type == "application/json":
        sequences = json.loads(request_body)
        logger.info(sequences)
        
        return encode_sequences(sequences)

    else:
        logger.exception(f"Unhandled content type in input fn: {request_content_type}")
        raise


def predict_fn(input_data, model):
    logger.info(f"predict_fn")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']
    
    input_ids.to(device)
    attention_mask.to(device)
    
    with torch.no_grad():
        return model(input_ids, attention_mask)

import datasets
from torch.utils.data import random_split, DataLoader
import os
import torch
import pandas as pd

from feature_vectors import create_audio_processor#feature_vector_function, load_audio
from SequentialSpectrogramPredictionDataset import SequentialSpectrogramPredictionDataset

# global dictionary that can map between the name of a dataset that we use and what might be used natively in 
# libraries such as torch or huggingface
DATASET_LOOKUP = {
    "TIMIT" : "timit_asr",
    "librispeech_asr" : "librispeech_asr"
}
#VOCALLBASE_LOOKUP = {
#    "PIHA"
#}
def get_data(experiment_parameters):
    if isinstance(experiment_parameters["dataset(s)"], str):
        return get_dataset(experiment_parameters)
    elif isinstance(experiment_parameters["dataset(s)"], list):
        return get_datasets(experiment_parameters)



def get_dataset(experiment_parameters):
    #waveform_fn = load_audio(experiment_parameters)
    #spectrogram_fn = feature_vector_function(experiment_parameters)
    audio_processor = create_audio_processor(experiment_parameters)
    full_data_path = os.path.join(experiment_parameters["data_path"], experiment_parameters["dataset(s)"])

    train_dataset = None
    test_dataset = None
    train_dataset_size = 0
    # this is for dataset built in to huggingface, typically used for human benchmarks
    # Handle HuggingFace hub datasets
    if experiment_parameters["dataset(s)"] in DATASET_LOOKUP.keys():
        dataset_name = DATASET_LOOKUP[experiment_parameters["dataset(s)"]]
        
        # Special handling for LibriSpeech
        if dataset_name == "librispeech_asr":
            # Load from cache (it's already downloaded to your specified location)
            train_dataset = datasets.load_dataset(
                "librispeech_asr", 
                "clean", 
                split="train.100",
                cache_dir=experiment_parameters["data_path"]
            )
            
            # For test set
            #if experiment_parameters["test_ratio"] == 0:
            test_dataset = datasets.load_dataset(
                "librispeech_asr", 
                "clean", 
                split="test",
                cache_dir=experiment_parameters["data_path"]
            )
            
            train_dataset_size = len(train_dataset)
            
        else:
            # Handle other HuggingFace datasets that might use data_dir (such as TIMIT)
            full_data_path = os.path.join(experiment_parameters["data_path"], experiment_parameters["dataset(s)"])
            base_dataset = datasets.load_dataset(dataset_name, data_dir=full_data_path)
            train_dataset = base_dataset['train']
            train_dataset_size = len(base_dataset['train'])
            
            #if experiment_parameters["test_ratio"] == 0:
            test_dataset = base_dataset['test']

#    if experiment_parameters["dataset(s)"] in DATASET_LOOKUP.keys():
#        base_dataset = datasets.load_dataset(DATASET_LOOKUP[experiment_parameters["dataset(s)"]], data_dir=full_data_path)

#        train_dataset = base_dataset['train']
#        train_dataset_size = len(base_dataset['train'])
#        # Case where we are using a baseline dataset that has a defined test set
#        if experiment_parameters["test_ratio"] == 0:
#            test_dataset = base_dataset['test']
#        else:
#            # will have to handle this train and test case separately
#            pass

    # Unless otherwise stated, assume that the vocallbase format is being used, typically used for animal vocalizations
    elif experiment_parameters["vocallbase_format"]:
        vocallbase_df = pd.read_csv(os.path.join(experiment_parameters["data_path"], experiment_parameters["dataset(s)"]+".csv"))
        train_dataset = datasets.Dataset.from_pandas(vocallbase_df[vocallbase_df["train"] == 1])
        test_dataset = datasets.Dataset.from_pandas(vocallbase_df[vocallbase_df["train"] == 0])

        train_dataset_size = len(train_dataset)


    test_dataset = SequentialSpectrogramPredictionDataset(experiment_parameters["dataset(s)"], test_dataset, 
        audio_processor, experiment_parameters["context_size"], full_data_path, non_overlap_offset=experiment_parameters["non_overlap_offset"], 
        target_sample_rate=experiment_parameters["sample_rate"], vocallbase_format=experiment_parameters["vocallbase_format"])

    train_dataset = SequentialSpectrogramPredictionDataset(experiment_parameters["dataset(s)"], train_dataset, 
            audio_processor, experiment_parameters["context_size"], full_data_path, non_overlap_offset=experiment_parameters["non_overlap_offset"], 
            target_sample_rate=experiment_parameters["sample_rate"], vocallbase_format=experiment_parameters["vocallbase_format"])
    # initializing output variables
    train_indices = None
    validation_indices = None
    #test_indices = None

    #train_dataset_size = len(base_dataset['train'])
    indices = torch.randperm(train_dataset_size)

    train_size = int((1-experiment_parameters["validation_ratio"]) * train_dataset_size)
    train_indices = indices[:train_size]

    if experiment_parameters["validation_ratio"] != 0:
        validation_size = int(experiment_parameters["validation_ratio"] * train_dataset_size)
        validation_indices = indices[train_size:train_size+validation_size]
    
    #if experiment_parameters["test_ratio"] != 0:
    #    test_size = train_dataset_size - train_size - validation_size  # Ensure no rounding issues
    #    test_indices = indices[train_size+validation_size:]

    
    return train_dataset, test_dataset, train_indices, validation_indices#, test_indices

    
    
# TODO: need to be able to handle the case where datasets are tested on independently AND aggregated.
def get_datasets(experiment_parameters):
    pass
    

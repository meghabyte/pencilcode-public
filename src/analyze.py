from datasets import load_from_disk
from collections import Counter
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datasets import concatenate_datasets, disable_progress_bar
import datasets
from src.utils import set_seed

def get_seen_data(train_data, checkpoint_step, batch_size, is_seen=True):
    """Helper function to get seen data from the training data and number of steps."""
   
    # print(f'Training file: {train_file}') 
    # train_data = load_from_disk(train_file)
    
    # convert checkpoint_step to index in the dataset
    checkpoint_index = checkpoint_step * batch_size
    print(f'Getting data for checkpoint step {checkpoint_step} and batch size {batch_size}: index {checkpoint_index}...')
    print(f'Data being indexed:')
    print(train_data)
    
    if is_seen:
        data_to_return = train_data.select(range(min(checkpoint_index, len(train_data))))
    else:
        if checkpoint_index >= len(train_data):
            print(f'Warning: checkpoint index {checkpoint_index} is greater than the length of the training data {len(train_data)}')
            return []
        else:
            data_to_return = train_data.select(range(checkpoint_index, len(train_data)))
    
    return data_to_return

def get_users_by_seen(train_data, checkpoint_step, batch_size, is_seen=True):
    """Helper function to get seen users from the training data and number of steps.
    
    Args:
        train_data (Dataset): Training data.
        checkpoint_step (int): Checkpoint step.
        batch_size (int): Batch size used to train model.
    """
   
    # get seen users from seen data, then get unseen users b ased on this 
    seen_data = get_seen_data(train_data, checkpoint_step, batch_size, is_seen=True)
    seen_users = seen_data["username_hashes"]
    
    seen_counter = Counter(seen_users)
    unique_seen_users = set(seen_users)
    
    
    num_total_unique_users = len(set(train_data["username_hashes"]))
    print(f'Number of unique seen users: {len(unique_seen_users)}/{num_total_unique_users}')
    
    if is_seen:
        print(f'Getting seen users...') 
        return unique_seen_users, seen_counter
   
    else:
        print(f'Getting unseen users...')
        # get unseen users
        unseen_users = set([u for u in set(train_data["username_hashes"]) if u not in unique_seen_users])
        assert len(unseen_users) + len(unique_seen_users) == num_total_unique_users, f'Expected unseen users + seen users to equal total number of users: {len(unseen_users)} + {len(unique_seen_users)} != {num_total_unique_users}'
        return unseen_users, None
    
def get_users_to_analyze(train_datas, student_source_data=None, seed=0, min_num_traces=50, max_num_traces=200, num_processes=50, num_users_to_sample=None, percent_users_to_sample=None, only_return_users=False):
    """Get users to analyze from the training data based on number of traces.
    # TODO: similar to create_user_finetuning_data.py, but with some differences.
    student_source_data: data used for getting user_to_data for computing ground truth metrics
    """
    
    set_seed(seed)
    
    all_user_counts = []
    for train_data in train_datas:
        print(train_data)
        user_counts = get_user_counts(train_data)
        temp_user_counts = {k: v for k, v in user_counts.items() if v >= min_num_traces and v <= max_num_traces}
        all_user_counts.append(temp_user_counts)
    
    # get intersection of users across all datasets
    all_user_counts = [set(user_counts.keys()) for user_counts in all_user_counts]
    users_to_sample_from = list(set.intersection(*all_user_counts))
    users_to_sample_from = list(sorted(users_to_sample_from))
        
    # get seen users from training data
    # user_counts = get_user_counts(train_data)
    # temp_user_counts = {k: v for k, v in user_counts.items() if v >= min_num_traces and v <= max_num_traces}
    # users_to_sample_from = list(temp_user_counts.keys())
    assert len(users_to_sample_from) > 0, f'No users with at least {min_num_traces} traces and at most {max_num_traces} traces'
    
    if percent_users_to_sample is not None:
        # get number of users to sample based on percentage
        print(f'Getting percentage of users to sample: {percent_users_to_sample}')
        num_users_to_sample = int(percent_users_to_sample * len(users_to_sample_from))
        print(f'Number of users to sample: {num_users_to_sample}/{len(users_to_sample_from)}')
    
    assert num_users_to_sample <= len(users_to_sample_from), f'Number of users to sample {num_users_to_sample} is greater than number of users to sample from {len(users_to_sample_from)}'
    print(f'Number of users with at least {min_num_traces} traces and at most {max_num_traces} traces: {len(users_to_sample_from)}')
     
    user_to_data = {}
    sampled_users = np.random.choice(users_to_sample_from, num_users_to_sample, replace=False)
    
    print(f'Number of sampled users: {len(sampled_users)}')
    
    if only_return_users:
        return sampled_users
    
    # filter all instances of dataset where username is not in users_to_sample_from (for speed)
    print(f'Filtering dataset to only include users in users_to_sample_from (with more than {min_num_traces} traces and at most {max_num_traces} traces)...')
    if student_source_data is None:
        assert len(train_datas) == 1, f'If student_source_data is None, then only one training data should be passed in. Got {len(train_datas)}'
        student_source_data = train_datas[0]
    dataset = student_source_data.filter(lambda x: x['username'] in users_to_sample_from, load_from_cache_file=True, num_proc=num_processes)
    
    print(f'len(dataset) after filtering to only include users in users_to_sample_from: {len(dataset)}')
    
    # breakpoint()
     
    # shuffle users_to_sample_from and iterate through until hitting args.num_unseen_users
    print(f'Gathering data for {len(sampled_users)} users...')
    pbar = tqdm(sampled_users, total=len(sampled_users))
    temp_user_idx = 0
    for user in pbar:
        disable_progress_bar()
        user_data = dataset.filter(lambda x: x['username'] == user, load_from_cache_file=True)
        # breakpoint() 
        user_to_data[str(user)] = user_data
        # every 10 users, filter out the users that have already been sampled for efficiency
        if temp_user_idx == 50:
            dataset = dataset.filter(lambda x: x['username'] not in sampled_users, num_proc=int(num_processes/10), load_from_cache_file=True)
            temp_user_idx = 0
    
    # create a dataset by adding user data for each user in selected_user_names
    print(f'Concatenating data for {len(sampled_users)} users...')
    sampled_user_data = concatenate_datasets([user_to_data[user] for user in sampled_users])
    print('Done.')
    print(f'Resulting data for users:')
    print(sampled_user_data)
    # for user, d in user_to_data.items():
    #     print(f'{user}: {len(d)}')
        
    print(f'Size of resulting dataset: {len(sampled_user_data)}')
    
    user_to_data = datasets.DatasetDict(user_to_data)

    
    return sampled_user_data, user_to_data

def get_user_counts(data):
    """Get user counts from data."""
    
    user_counts = Counter(data["username"])
    return user_counts
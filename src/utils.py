import urllib
from collections import defaultdict, Counter
import os
import h5py
import numpy as np
import json
import re
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import torch
import bisect
import sqlite3
import random
from argparse import Namespace
import scipy.stats as stats
from src.data.split import read_splits
from src.global_vars import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from datetime import datetime, timedelta


CHECKPOINTS_DICT = {"all":f'final_models/all-checkpoint-440000',
                    "all_downsampled":f'final_models/all_downsampled-checkpoint-226000',
                    "last":"final_models/last-262000",
                    "synthetic":"final_models/synthetic-checkpoint-523000",
                    "synthetic_downsampled":"final_models/synthetic_downsampled-300000",
                  "all_1":f'final_models/all-checkpoint-100000',
                   "all_2":f'final_models/all-checkpoint-183000',
                   "all_3":f'final_models/all-checkpoint-300000',
                   "all_4":f'final_models/all-checkpoint-350000',
                   "all_5":f'final_models/all-checkpoint-400000',
                   "all_6":f'final_models/all-checkpoint-440000'}

def pearsonr(y, y_pred):
    return stats.pearsonr(y, y_pred)[0]


def add_minutes(ts: str, minutes: float, fmt="%Y-%m-%d %H:%M:%S") -> str:
    return (datetime.strptime(ts, fmt) + timedelta(minutes=minutes)).strftime(fmt)

def get_model_tokenizer(checkpoint_dir):
    from transformers import AutoTokenizer
    from src.student import StudentModel
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    print("Loading Model")
    model = StudentModel.from_pretrained(checkpoint_dir)
    model.to(device)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 
                                  'mask_token': '<mask>',
                                  'additional_special_tokens': ['<start>']})
    print("Loaded student model!")
    return model, tokenizer

def print_dict(d, keys_to_skip=None):
    if isinstance(d, Namespace):
        temp_d = d.__dict__
    elif isinstance(d, dict):
        temp_d = d.copy()
    else:
        raise ValueError(f'Invalid type: {type(d)}')
    if keys_to_skip is not None:
        # pop
        for key in keys_to_skip:
            temp_d.pop(key)
    # pretty print json
    print(json.dumps(temp_d, indent=4))
    
def get_broken_index_property(values):
    N = next((int(v.split("_")[-1])                # grab the piece after the last “_”
     for v in values
     if v.startswith("broken_index_") and v.split("_")[-1].isdigit()),
    None
    )
    return N
    
def write_success(out_dir):
    print(f'Writing "SUCCESS" to {out_dir}')
    # write "SUCCESS" to out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, "SUCCESS"), "w") as f:
        f.write("SUCCESS")
        
def check_success(out_dir):
    if os.path.exists(os.path.join(out_dir, "SUCCESS")):
        with open(os.path.join(out_dir, "SUCCESS"), "r") as f:
            if f.read() == "SUCCESS":
                return True
    return False

def load_tokenizer(checkpoint_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 
                                'mask_token': '<mask>',
                                'additional_special_tokens': ['<start>']})
    return tokenizer

def load_model(checkpoint_dir, device='cpu'):
    from src.student import StudentModel
    model = StudentModel.from_pretrained(checkpoint_dir)
    tokenizer = load_tokenizer(checkpoint_dir)
    
    print(f"Loaded student model from {checkpoint_dir} to {device}")
    model.to(device)
    return model, tokenizer

def load_db_connection(file_name='./data/splits/username_mappings.db'):
    # Open database connection
    conn = sqlite3.connect(file_name)
    cursor = conn.cursor()
    return conn, cursor

timestamp_pattern = r"\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\]"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class UsernameHash():
    def __init__(self, data_dir = './data'):
        self.unique_ids = {}
        self.conn, self.cursor = load_db_connection(os.path.join(data_dir, 'splits/username_mappings.db'))
        self.hash_to_username = json.load(open(os.path.join(data_dir, f'splits/hash_to_username_mapping.json'), 'r'))
        
    def get_hash(self, username):
        self.cursor.execute('SELECT user_id FROM mappings WHERE username = ?', (username,))
        result = self.cursor.fetchone()
        if not result:
            print(f'Unrecognized username: {username}')
            breakpoint()
        output = result[0] if result else MAX_NUM_USERS-1
        return output
    
    def get_username(self, hash):
        return self.hash_to_username[str(hash)]
    
    def get_sorted_usernames(self):
        return sorted(self.hash_to_username.values())
    
    def get_sorted_hashes(self):
        return sorted(self.hash_to_username.keys())


# TODO: what happens if user has multiple files with same name? (is this possible)
def get_solutions_dict(outer_key="username", parse_states=False):
    print("Reading logs...")
    solutions_dict = defaultdict(lambda: defaultdict(list))
    with open("sample.txt", "r") as f:
        lines = f.readlines()
    for l in lines:
        if "run&" in l:
            get_request = l[l.index("/log") :].split(" ")[0]
            username = l.split(" ")[1].lower()
            task = get_request.split("?")[0].replace("/log/", "").lower()
            if parse_states:
                get_request = parse_state(get_request)
            if outer_key == "username":
                if (
                    len(solutions_dict[username][task]) == 0
                    or get_request != solutions_dict[username][task][-1]
                ):
                    solutions_dict[username][task].append(get_request)
            elif outer_key == "task":
                if (
                    len(solutions_dict[task][username]) == 0
                    or get_request != solutions_dict[task][username][-1]
                ):
                    solutions_dict[task][username].append(get_request)
            else:
                raise ValueError(outer_key)
    return solutions_dict


# def parse_state(state_str):
#     parsed_lines = ""
#     code_lines = state_str.split("&")[-1].replace("code=","").split("|")
#     for line in code_lines:
#         parsed_lines += urllib.parse.unquote(line.replace("+"," "))+"\n"
#     return parsed_lines+"\n\n"


def print_mean_std(x):
    mean = np.mean(x)
    stdev = np.std(x)
    print(f"Mean: {mean:.6f}, Stdev: {stdev:.6f}")
    
def parse_state(state_str):
    # More efficient version of the above function
    # TODO: make sure they are equivalent (deleted the end "\n\n" in the above function)
    return urllib.parse.unquote(
        state_str.split("&")[-1]
        .replace("code=", "")
        .replace("|", "\n")
        .replace("+", " ")
    ).strip()


def create_trace_std(trace_log_dir="trace_logs"):
    lines_dict = defaultdict(lambda: defaultdict(list))
    count = 0
    program_count = 0
    for trace_log in os.listdir(trace_log_dir):
        date = trace_log.split("-")[1]
        with open(trace_log_dir + "/" + trace_log, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        # breakpoint()
        try:
            for l in lines:
                try:
                    user_id = l.split(" ")[1]
                    program_name = l.split("?")[0].split("/")[-1].lower().strip()

                    # get time by matching things like [04/Feb/2022:06:25:26 +0000]
                    match = re.search(timestamp_pattern, l)
                    if match:
                        timestamp_str = match.group(1)
                        dt = datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
                        # breakpoint()
                        # format date nicely, including year, month, day, hour, minute, second
                        time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        time = None

                    if (
                        len(program_name) > 100 or len(program_name) == 0
                    ):  # skip corrupted logs
                        continue
                    program_code = l.split("code=")[1].split(" ")[0]
                    lines_dict[user_id][program_name].append((program_code, date, time))
                    count += 1

                except:
                    continue
        except:
            continue
        if count > 100000:
            for uid in lines_dict.keys():
                for program_name in lines_dict[uid].keys():
                    program_count += 1
                    if not os.path.exists(trace_log_dir + "_std/" + uid[0]):
                        os.makedirs(trace_log_dir + "_std/" + uid[0])
                    if not os.path.exists(
                        trace_log_dir + "_std/" + uid[0] + "/" + uid[:3]
                    ):
                        os.makedirs(trace_log_dir + "_std/" + uid[0] + "/" + uid[:3])
                    if not os.path.exists(
                        trace_log_dir + "_std/" + uid[0] + "/" + uid[:3] + "/" + uid[:5]
                    ):
                        os.makedirs(
                            trace_log_dir
                            + "_std/"
                            + uid[0]
                            + "/"
                            + uid[:3]
                            + "/"
                            + uid[:5]
                        )
                    if not os.path.exists(
                        trace_log_dir
                        + "_std/"
                        + uid[0]
                        + "/"
                        + uid[:3]
                        + "/"
                        + uid[:5]
                        + "/"
                        + uid
                    ):
                        os.makedirs(
                            trace_log_dir
                            + "_std/"
                            + uid[0]
                            + "/"
                            + uid[:3]
                            + "/"
                            + uid[:5]
                            + "/"
                            + uid
                        )
                    fn = (
                        trace_log_dir
                        + "_std/"
                        + uid[0]
                        + "/"
                        + uid[:3]
                        + "/"
                        + uid[:5]
                        + "/"
                        + uid
                        + "/"
                        + program_name
                    )
                    with open(fn, "a") as f:
                        for program_code, date, time in lines_dict[uid][program_name]:
                            f.write(f"{program_code}, DATE: {date}, TIME: {time}")
                            f.write("\n")
            print(program_count)
            count = 0
            lines_dict = defaultdict(lambda: defaultdict(list))


def write_all_data(trace_log_dir="trace_logs"):
    # all_data includes user_id, program_name, program_code, date, time across all logs (huge file)
    all_data_file_name = "all_data.txt"
    print(f'Writing all data to "{all_data_file_name}"...')
    all_data_file = open(all_data_file_name, "w")
    all_data = []

    count = 0
    program_count = 0

    logs = os.listdir(trace_log_dir)
    num_logs = len(logs)

    for trace_log in tqdm(logs, total=num_logs):
        # if count > 10000000:
        #     break

        date = trace_log.split("-")[1]
        with open(trace_log_dir + "/" + trace_log, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue

        try:
            for l in lines:
                try:
                    user_id = l.split(" ")[1]
                    program_name = l.split("?")[0].split("/")[-1].lower().strip()

                    # get time by matching things like [04/Feb/2022:06:25:26 +0000]
                    match = re.search(timestamp_pattern, l)
                    if match:
                        timestamp_str = match.group(1)
                        dt = datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
                        # breakpoint()
                        # format date nicely, including year, month, day, hour, minute, second
                        time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        time = None

                    if (
                        len(program_name) > 100 or len(program_name) == 0
                    ):  # skip corrupted logs
                        continue
                    program_code = l.split("code=")[1].split(" ")[0]

                    all_data.append((user_id, program_name, program_code, date, time))
                    count += 1

                except:
                    continue
        except:
            continue

        # if all_data is more than 1000, write to file
        if len(all_data) > 1000:
            for d in all_data:
                # write each d separated by "\t"
                all_data_file.write("\t".join(d))
                all_data_file.write("\n")
            all_data = []

    # write remaining all_data
    for d in all_data:
        all_data_file.write("\t".join(d))
        all_data_file.write("\n")
    # close
    all_data_file.close()


def get_stats(d="trace_logs_std"):
    num_users = 0
    program_names = []
    for f in os.listdir(d):
        for uid in os.listdir(d + "/" + f):
            num_users += 1
            for pn in os.listdir(d + "/" + f + "/" + uid):
                program_names.append(pn)
        print(len(program_names))
    print("/n")
    print(Counter(program_names))
    print(num_users)
    print(len(set(program_names)))


def process_file(file_path, version_date_format="int"):
    """Parses a file and returns a list of tuples (username, program name, program code, version date)."""
    data = []
    with open(file_path, "r") as f:
        for l in f.readlines():
            try:
                username = l.split(" ")[1]
                program_name = l.split("?")[0].split("/")[-1].lower().strip()
                if (
                    len(program_name) > 100 or len(program_name) == 0
                ):  # skip corrupted logs
                    continue
                program_code = l.split("code=")[1].split(" ")[0]
                match = re.search(timestamp_pattern, l)
                if match:
                    timestamp_str = match.group(1)
                    dt = datetime.strptime(timestamp_str, "%d/%b/%Y:%H:%M:%S %z")
                    if version_date_format == "int":
                        version_date = int(dt.timestamp())
                    elif version_date_format == "str":
                        version_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        raise ValueError(
                            f"Invalid version_date_format: {version_date_format}"
                        )
                else:
                    continue
                data.append((username, program_name, program_code, version_date))
            except:
                continue
    return data


def extract_year_from_log(log_filename: str) -> int:
    # Split the filename by dashes or underscores to isolate the date
    parts = log_filename.split("-")
    # The part containing the date is usually the second element
    date_part = parts[1]
    # Extract the year (first 4 digits from the date)
    year = date_part[:4]
    return str(year)


def write_batch_to_hdf5(hdf5_file, batch_data):
    """Writes a batch of data to the HDF5 file."""
    compound_dtype = np.dtype(
        [("string", h5py.special_dtype(vlen=str)), ("integer", "i4")]
    )
    for username, programs in batch_data.items():
        if username not in hdf5_file:
            user_group = hdf5_file.create_group(username)
        else:
            user_group = hdf5_file[username]
        for program_name, program_entries in programs.items():
            program_entries_np = np.array(program_entries, dtype=compound_dtype)
            if program_name not in user_group:
                user_group.create_dataset(
                    program_name,
                    data=program_entries_np,
                    maxshape=(None,),
                    chunks=True,
                    dtype=compound_dtype,
                )
            else:
                dataset = user_group[program_name]
                new_size = dataset.shape[0] + len(program_entries)
                dataset.resize((new_size,))
                dataset[-len(program_entries) :] = program_entries


def create_hdf5_from_files_in_batches(hdf5_path):
    """Processes multiple files in batches and stores the results in an HDF5 file."""
    batch_data = {}
    i = 0
    file_count = 0
    for file_path in os.listdir("trace_logs/"):
        year = extract_year_from_log(file_path)
        with h5py.File(hdf5_path + "_" + year + ".h5", "a") as hdf5_file:
            print(file_count)
            file_data = process_file("trace_logs/" + file_path)
            for username, program_name, program_code, version_date in file_data:
                if username not in batch_data:
                    batch_data[username] = {}
                if program_name not in batch_data[username]:
                    batch_data[username][program_name] = []
                batch_data[username][program_name].append((program_code, version_date))
                i += 1
                if (i + 1) % 10000 == 0:
                    write_batch_to_hdf5(hdf5_file, batch_data)
                    batch_data.clear()
                    print(f"Processed {i + 1} programs...")
            file_count += 1
    if batch_data:
        with h5py.File(hdf5_path + "_" + year + ".h5", "a") as hdf5_file:
            write_batch_to_hdf5(hdf5_file, batch_data)
            print("Final batch written.")


def get_id_name_counts(
    out_file="id_name_counts.txt", do_overwrite=False, trace_logs_dir="trace_logs"
):
    """Helper function to get a counter of # programs per user_id/program_name"""

    years = set()
    if do_overwrite or not os.path.exists(out_file):
        id_name_counts = Counter()
        files = os.listdir(trace_logs_dir)
        files.sort()
        for file_path in tqdm(files):
            year = extract_year_from_log(file_path)
            years.add(year)
            file_data = process_file(f"{trace_logs_dir}/{file_path}")
            for username, program_name, program_code, version_date in file_data:
                id_name_counts[(username, program_name)] += 1
                # breakpoint()

        # print sorted years
        print(f"Years: {sorted(years)}")
        # write as dict
        print(f"Writing id_name_counts to {out_file}...")
        # can't write tuples to json, so write manually
        with open(out_file, "w") as f:
            # write ordered
            for (username, program_name), count in id_name_counts.items():
                f.write(f"{username}\t{program_name}\t {count}\n")

    print(f"Reading id_name_counts from {out_file}...")
    id_name_counts = pd.read_csv(
        out_file, sep="\t", header=None, names=["user_id", "program_name", "count"]
    )

    num_pairs = len(id_name_counts)

    # print formatted in millions
    num_unique_users = len(id_name_counts["user_id"].unique())
    print(f'Total unique usernames: {num_unique_users:,}')
    print(f"Total unique (username, program_name) pairs: {num_pairs:,}")
    print(f'Average number of program names per user: {num_pairs / num_unique_users:.2f}')
    print(f'Total programs: {id_name_counts["count"].sum():,}')

    return id_name_counts


# hdf5_path = 'output.h5'
# create_hdf5_from_files_in_batches('output')


def create_csv_from_files(
    split_dir="splits", out_dir="csv_files", sep="\t", trace_logs_dir="trace_logs", seed=0
):
    """Processes multiple files in batches and stores the results in a CSV file.
    - Reads the splits from the "splits" directory
    - Writes the data to CSV files in the "out_dir" directory, writing to a separate file for each split.
    """
    print("Creating CSV files from trace logs/split tuples...")
    batch_data = {}
    i = 0
    file_count = 0
    splits = read_splits(split_dir=split_dir)
    
    set_seed(seed)

    # print len(splits)
    print("Read splits:")
    for split_name, split_data in splits.items():
        print(f"- {split_name}:\t{len(split_data):,} user-program pairs")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    split_to_csv = {}
    for split_name in splits.keys():
        csv_file_name = f"{out_dir}/{split_name}.csv"
        csv_file = open(csv_file_name, "w")
        csv_file.write(f"username{sep}program_name{sep}program_code{sep}version_date\n")
        split_to_csv[split_name] = csv_file

    split_counts = {split_name: 0 for split_name in splits.keys()}

    print("Iterating through data in trace_logs...")
    file_paths = os.listdir(trace_logs_dir)
    file_paths.sort()
    for file_path in tqdm(file_paths):
        file_data = process_file(
            f"{trace_logs_dir}/{file_path}", version_date_format="str"
        )
        # print(file_path)
        for username, program_name, program_code, version_date in file_data:
            for split_name, split_data in sorted(splits.items()):
                if (username, program_name) in split_data:
                    csv_file = split_to_csv[split_name]
                    csv_file.write(
                        f"{username}{sep}{program_name}{sep}{program_code}{sep}{version_date}\n"
                    )

                    split_counts[split_name] += 1
                # i += 1
                # if (i + 1) % 10000 == 0:
                # print(f"Processed {i + 1} programs...")

    for split_name, csv_file in split_to_csv.items():
        csv_file.close()
        print(
            f'Wrote {split_counts[split_name]} programs to "{out_dir}/{split_name}.csv"'
        )

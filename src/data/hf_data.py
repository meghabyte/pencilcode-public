from collections import Counter
import numpy as np
import pandas as pd
import os
import time
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from collections import defaultdict

"""Script for writing preprocessed data to disk"""

from src.data.split import *
from src.utils import parse_state

def create_synthetic_trace(last_program, target_len):
        split_program = last_program.split("|")
        build_program = []
        for i in range(1, len(split_program)-1):
            build_program.append("|".join(split_program[0:i])+"|")
        if(target_len < len(split_program)):
            return build_program[-target_len:]
        else:
            return build_program
        
def preprocess_dataset(dataset, do_parse_state=True, data_type="all"):
    """
    Preprocesses a HuggingFace dataset to create a structured format where each
    user_id/program_name pair has a chronologically ordered list of program codes.

    Args:
        dataset: HuggingFace dataset with columns user_id, program_name, program_code, timestamp
        do_parse_state: whether to parse the state, i.e. convert the state to a more readable format (default: True)
        data_type: whether to take all program codes or just the last one in formatting 'program_text'

    Returns:
        HuggingFace dataset with columns user_id, program_name, program_codes, version_dates
    """

    valid_data_types = ["all", "last", "synthetic"]
    if data_type not in valid_data_types:
        raise ValueError(f"data_type must be one of {valid_data_types}")

    df = dataset.to_pandas()

    # Sort by timestamp within each group
    print("Sorting by user_id, program_name, and timestamp...")
    df = df.sort_values(["username", "program_name", "version_date"])
    print("Done.")

    def map_code(code, do_parse_state=True):
        if do_parse_state:
            return "" if code is None else parse_state(code)
        return "" if code is None else code

    def format_program_codes_str(row, do_parse_state=True):
        # skip duplicates
        
        # first remove duplicates
        non_duplicates = []
        for (code, date, prev_code) in zip(row["program_codes"], row["version_dates"], [None]+row["program_codes"][:-1]):
            if map_code(code, do_parse_state=do_parse_state) != map_code(prev_code, do_parse_state=do_parse_state):
                non_duplicates.append((code, date))
            # else:
            #     print(f"Duplicate code: {code}")
            #     print(f"Previous code: {prev_code}")
                
        program_codes = [
            f"CODE {i+1} ({date}):\n{map_code(code, do_parse_state=do_parse_state)}"
            for i, (code, date) in enumerate(non_duplicates)
        ]
        
        # if '|' in row['program_codes']:
        #     breakpoint()
        
        # program_codes = [
        #         f"CODE {i+1} ({date}):\n{map_code(code, do_parse_state=do_parse_state)}"
        #         for i, (code, date, prev_code) in enumerate(
        #             zip(row["program_codes"], row["version_dates"], [None]+row["program_codes"][:-1])
        #         )
        #         if map_code(code, do_parse_state=do_parse_state) != map_code(prev_code, do_parse_state=do_parse_state)
        #     ]
        
        program_code_str = "\n".join(program_codes)
        # return f'USER: {row["username"]}; PROGRAM: {row["program_name"]}; {program_code_str}'
        # TODO: hacky, but adding <|endoftext|> to the end of each program code here
        code = f"{program_code_str.strip()}<|endoftext|>"
        if code == "<|endoftext|>":
            print(f"Empty code: {row['username']}, {row['program_name']}")
            print('original program codes:', row['program_codes'])
            print('non duplicates:', non_duplicates)
            # should only happen if all program codes are None or |
            if not all([c in [None, '|', '%09|', '||', '+|', '|++|++|', '|++||', '|++|'] for c in row['program_codes']]):
                breakpoint()
            # assert all([c is None or c =='|' for c in row['program_codes']]), f'All program codes are not None or |: {row["program_codes"]}'
            
        # if '|' in row['program_codes'] or None in row['program_codes']:
        #     breakpoint()
        return code

    def get_prefix(row):
        return f'{row["username"]+" "+row["program_name"]+" "}'

    print("Grouping by user_id and program_name...")
    # Group by user_id and program_name and aggregate the program codes and timestamps
    grouped_df = (
        df.groupby(["username", "program_name"])
        .agg(
            {
                "program_code": list,
                # 'version_date': lambda x: [t.strftime('%Y-%m-%d %H:%M:%S') for t in x]
                "version_date": list,
            }
        )
        .reset_index()
    )

    # Make sure the version dates are increasing
    print("Checking that version dates are increasing for a random sample...")
    samples = grouped_df.sample(5)
    for _, row in samples.iterrows():
        assert all(
            [
                orig == s
                for (orig, s) in zip(row["version_date"], sorted(row["version_date"]))
            ]
        ), f"Version dates are not increasing for {row['username']}, {row['program_name']}: {row['version_date']}"
    print("Done.")

    if data_type == "last":
        print("Getting the last program code and timestamp...")
        # Get the last program code and timestamp
        grouped_df["program_code"] = grouped_df["program_code"].apply(lambda x: [x[-1]])
        grouped_df["version_date"] = grouped_df["version_date"].apply(lambda x: [x[-1]])
    elif data_type == "synthetic":
        print("Getting synthetic program code and timestamp...")
        # Get the last program code and timestamp
        pd.set_option('display.max_colwidth', None)
        print(grouped_df.sample(n=1)["program_code"])
        print(grouped_df.sample(n=1)["version_date"])
        grouped_df["program_code"] = grouped_df["program_code"].apply(lambda x: create_synthetic_trace(x[-1], min(len(x), len(x[-1].split("|")))) if (x is not None and x[-1] is not None) else x)
        grouped_df["version_date"] = grouped_df["version_date"].apply(lambda x: x[-min(len(x[-1].split("|")), len(x)):] if x is not None and isinstance(x, str) else x)
        print(grouped_df.sample(n=3)["program_code"])
        print(grouped_df.sample(n=3)["version_date"])

    # Rename the timestamp column to version_dates
    grouped_df = grouped_df.rename(
        columns={"program_code": "program_codes", "version_date": "version_dates"}
    )

    # Format the program codes as a string; use multiprocessing for speed
    print("Formatting program codes...")
    tqdm.pandas()
    grouped_df["program_text"] = grouped_df.progress_apply(
        lambda row: format_program_codes_str(row, do_parse_state=do_parse_state), axis=1
    )
    print(f'Before dropping empty program codes: {len(grouped_df)}')
    # drop empty program codes
    grouped_df = grouped_df[grouped_df["program_text"] != "<|endoftext|>"]
    print(f'After dropping empty program codes: {len(grouped_df)}')

    # drop the program_codes and version_dates columns (to reduce size)
    grouped_df = grouped_df.drop(columns=["program_codes", "version_dates"])
    print("Done.")

    # Convert back to HuggingFace dataset
    print("Converting to HuggingFace dataset...")
    dataset = Dataset.from_pandas(grouped_df)
    print("Done.")
    return dataset


def load_preprocessed_data(data_dir="preprocessed_data"):
    # TODO: load unseen splits?
    data_files = {
        "train": f"{data_dir}/train.hf",
        "validation": f"{data_dir}/validation.hf",
        # "test": f"{data_dir}/test.hf",
        # "heldout_user_test": f"{data_dir}/heldout_user_test.hf",
        # "heldout_program_test": f"{data_dir}/heldout_program_test.hf",
    }

    dataset = DatasetDict()
    for split, file in data_files.items():
        try:
            dataset[split] = load_from_disk(file)
        except FileNotFoundError:
            print(f"File {file} not found. Skipping {split} split.")
            continue

    return dataset


# TODO: some of this data will have None (also for usernames?)
def write_preprocessed_data(
    split_data_dir="csv_files",
    output_dir="preprocessed_data",
    do_parse_state=True,
    data_type="all",
):
    data_files = {
        "train": f"{split_data_dir}/train.csv",
        # "test": f"{split_data_dir}/test.csv",
        "seen_user_seen_program": f"{split_data_dir}/seen_user_seen_program.csv",
        "seen_user_unseen_program": f"{split_data_dir}/seen_user_unseen_program.csv",
        "unseen_user_seen_program": f"{split_data_dir}/unseen_user_seen_program.csv",
        "unseen_user_unseen_program": f"{split_data_dir}/unseen_user_unseen_program.csv",
        # "heldout_user_test": f"{split_data_dir}/heldout_user_test.csv",
        # "heldout_program_test": f"{split_data_dir}/heldout_program_test.csv",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocessed_data = {}
    for split in [
        "train",
        # "test", 
        # "heldout_user_test", 
        # "heldout_program_test",
        "seen_user_seen_program",
        "seen_user_unseen_program",
        "unseen_user_seen_program",
        "unseen_user_unseen_program",
        ]:
        print(f"Preprocessing {split}...")
        start = time.time()

        # Only take the first 1000 rows -- used for testing
        # dataset[split] = dataset[split].select(range(1000))

        preprocessed_data[split] = preprocess_dataset(
            dataset[split], do_parse_state=do_parse_state, data_type=data_type
        )

        # BELOW IS FOR SANITY CHECKING THAT THE VERSION DATES ARE INCREASING
        # df = preprocessed_data[split].to_pandas()
        # samples = df.sample(5)
        # # make sure the version dates are increasing
        # for _, row in samples.iterrows():
        #     assert all([orig == s for (orig, s) in zip(row['version_dates'], sorted(row['version_dates']))])

        # Save the preprocessed dataset
        preprocessed_out_file = f"{output_dir}/{split}.hf"
        print(f"Saving preprocessed {split} to {preprocessed_out_file}...")
        preprocessed_data[split].save_to_disk(f"{preprocessed_out_file}")
        print(f"Done.")

        end = time.time()
        mins = (end - start) / 60
        # print rounded
        print(f"Time taken: {mins:.2f} minutes")

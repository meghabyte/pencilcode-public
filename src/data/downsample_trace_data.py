from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

from src.data.hf_data import load_preprocessed_data

"""A script for generating total # of tokens in dataset after tokenization. Does NOT include mask tokens for prefixes."""


def format_num_tokens(num_tokens):
    """Format number of tokens in billions."""
    return f"{num_tokens/1e9:.4f}"


def downsample_to_num_tokens(
    dataset, target_num_tokens, split="train", target_percent_diff=0.001, seed=0
):
    """Given a dataset, downsample to a certain number of tokens for the given split."""
    # get the number of tokens in the dataset
    num_tokens_in_dataset = sum(dataset[split]["length"])

    # get the ratio of the number of tokens in the dataset to the number of tokens we want
    ratio = target_num_tokens / num_tokens_in_dataset

    orig_dataset = dataset.copy()

    # print nicely formatted
    print(f"Target number of tokens: {format_num_tokens(target_num_tokens)}")

    print(f"Starting ratio: {ratio:.4f}")
    dataset["train"] = orig_dataset[split].train_test_split(
        ratio, seed=seed, shuffle=True
    )["test"]

    # downsample the dataset by sampling until we reach the target number of tokens; may require multiple passes; do binary search over number of examples to sample, starting at ratio
    while True:
        # get the number of tokens in the downsampled dataset
        num_tokens_in_downsampled_dataset = sum(dataset[split]["length"])
        # print nicely formatted in billions
        print(
            f"Number of tokens in downsampled dataset: {format_num_tokens(num_tokens_in_downsampled_dataset)}"
        )

        percent_diff = (
            abs(num_tokens_in_downsampled_dataset - target_num_tokens)
            / target_num_tokens
        )

        # if we are within 1% of the target, we are done
        if percent_diff < target_percent_diff:
            break

        # if we are under the target, we need to sample more
        if num_tokens_in_downsampled_dataset < target_num_tokens:
            ratio *= 1 + percent_diff / 2
        else:
            ratio *= 1 - percent_diff / 2

        print(f"New ratio: {ratio:.4f}")

        # sample the dataset
        dataset["train"] = orig_dataset[split].train_test_split(
            ratio, seed=seed, shuffle=True
        )["test"]

    print()
    print(
        f"Final number of tokens in downsampled dataset: {num_tokens_in_downsampled_dataset/1e9:.4f}"
    )
    print(f"Target number of tokens: {format_num_tokens(target_num_tokens)}")
    print(f"Final ratio: {num_tokens_in_downsampled_dataset/target_num_tokens:.4f}")
    percent_diff = (
        abs(num_tokens_in_downsampled_dataset - target_num_tokens) / target_num_tokens
    )
    print(f"Percent difference: {percent_diff:.4f}")

    return dataset


def run_downsampling(
    input_dir="preprocessed_data_parsed",
    out_dir="preprocessed_data_parsed/all_downsampled",
    preprocessing_num_proc=200,
    seed=0,
    source_data_type="all",
    data_types=["all", "last", "synthetic"],
):
    """Runs downsampling of trace data to match number of tokens in 'last' data.

    Args:
    - out_dir: directory to write downsampled trace data to

    Works by:
    - Tokenizing both types of data
    - Getting total number of tokens in each split across both data types
    - Downsampling 'all' data to match 'last' data
    - Writing downsampled data to disk in a new directory
    """

    total_tokens_by_data_type = {}

    tokenized_datasets_by_data_type = {}

    for data_type in data_types:
        preprocessed_data_dir = f"{input_dir}/{data_type}"

        raw_datasets = load_preprocessed_data(preprocessed_data_dir)
        print(raw_datasets)

        # load tokenizer and model, make sure to use config from checkpoint
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # TODO: figure out how to do this automatically based on the directory?
        tokenizer.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})

        def tokenize_function(examples):
            return tokenizer(
                examples["program_text"],
                padding="max_length",
                truncation=False,
                return_length=True,
            )

        # tokenize all the data and then get total number of tokens
        # take advantage of map for parallel processing
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_proc,
            load_from_cache_file=True,
            # num_proc=None,
            # remove_columns=raw_datasets.column_names
        )

        tokenized_datasets_by_data_type[data_type] = tokenized_datasets

        print(f"Total tokens in billions:")

        total_tokens_by_split = {}

        # get total number of tokens
        for split in tokenized_datasets:
            total_tokens = sum(tokenized_datasets[split]["length"])
            print(f" - {split}: {format_num_tokens(total_tokens)}")

            total_tokens_by_split[split] = total_tokens

        total_tokens_by_data_type[data_type] = total_tokens_by_split

    # print total tokens for each data type
    print("Total tokens (billions)")
    for dtype in total_tokens_by_data_type:
        print(f"{dtype.upper()}:")
        for split in total_tokens_by_data_type[dtype]:
            print(f" - {split}: {total_tokens_by_data_type[dtype][split]/1e9:.2f}")
        print()

    # downsample all 'train' to match 'last' 'train'
    target_num_tokens = total_tokens_by_data_type["last"]["train"]
    print(f"Target number of tokens: {format_num_tokens(target_num_tokens)}")

    # downsample all train
    downsampled_train = downsample_to_num_tokens(
        tokenized_datasets_by_data_type[source_data_type],
        target_num_tokens,
        split="train",
        seed=seed,
    )

    downsampled_all = tokenized_datasets_by_data_type[source_data_type]
    downsampled_all["train"] = downsampled_train["train"]

    # get total number of tokens after downsampling
    print(f"Total tokens in billions after downsampling:")
    for split in downsampled_all:
        total_tokens = sum(downsampled_all[split]["length"])
        print(f" - {split}: {total_tokens/1e9:.2f}")

    # write each split
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for split in downsampled_all:
        out_file = f"{out_dir}/{split}.hf"
        print(f"Writing {split} to {out_file}...")
        # Keep only original columns
        columns_to_keep = raw_datasets[split].column_names
        columns_to_remove = [
            col
            for col in downsampled_all[split].column_names
            if col not in columns_to_keep
        ]
        downsampled_all[split] = downsampled_all[split].remove_columns(
            columns_to_remove
        )
        downsampled_all[split].save_to_disk(out_file)

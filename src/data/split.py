from collections import Counter
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


def create_train_test_splits(
    id_name_counts,
    test_frac=0.2,
    frac_user_ids_to_holdout=0.03,
    frac_program_names_to_holdout=0.03,
    seed=0,
):
    """
    Returns a mapping from (user_id, program_name) to split (train, unseen_user_seen_program, unseen_user_unseen_program, seen_user_seen_program, seen_user_unseen_program).
    """

    import numpy as np

    np.random.seed(seed)

    tqdm.pandas()

    # need to format the key like this (instead of just concatenating user_id and program_name) because the same user_id and program_name can appear multiple times if, say, a similar user has written the program_name with '_'?
    # used for checking for duplicates
    id_name_counts["key"] = id_name_counts.progress_apply(
        lambda row: f"user_id={row['user_id']}_program_name={row['program_name']}",
        axis=1,
    )
    # make sure no duplicates
    assert len(id_name_counts) == len(
        id_name_counts["key"].unique()
    ), f'duplicates in id_name_counts: {len(id_name_counts) - len(id_name_counts["key"].unique())}'

    # get unique user_ids and program_names
    user_ids = id_name_counts["user_id"].unique()
    program_names = id_name_counts["program_name"].unique()

    # randomly select a fraction of user_ids and program_names to hold out
    user_ids_holdout = np.random.choice(
        user_ids, int(frac_user_ids_to_holdout * len(user_ids)), replace=False
    )
    program_names_holdout = np.random.choice(
        program_names,
        int(frac_program_names_to_holdout * len(program_names)),
        replace=False,
    )

    non_heldout = id_name_counts[
        ~id_name_counts["user_id"].isin(user_ids_holdout)
        & ~id_name_counts["program_name"].isin(program_names_holdout)
    ]

    train, _ = train_test_split(non_heldout, test_size=test_frac, random_state=seed)

    seen_users = train["user_id"].unique()
    seen_programs = train["program_name"].unique()
    
    unseen_user_seen_program = id_name_counts[
        ~id_name_counts["user_id"].isin(seen_users)
        & id_name_counts["program_name"].isin(seen_programs)
    ]
    unseen_user_unseen_program = id_name_counts[
        ~id_name_counts["user_id"].isin(seen_users)
        & ~id_name_counts["program_name"].isin(seen_programs)
    ]
    # make sure seen_user_seen_program doesn't have any keys that are in train keys
    seen_user_seen_program = id_name_counts[
        id_name_counts["user_id"].isin(seen_users)
        & id_name_counts["program_name"].isin(seen_programs)
        & ~id_name_counts["key"].isin(train["key"])
    ]
    seen_user_unseen_program = id_name_counts[
        id_name_counts["user_id"].isin(seen_users)
        & ~id_name_counts["program_name"].isin(seen_programs)
    ]
    
    num_splits_total = len(unseen_user_seen_program) + len(unseen_user_unseen_program) + len(seen_user_seen_program) + len(seen_user_unseen_program)

    assert len(id_name_counts) == num_splits_total + len(train), f"Expected len(id_name_counts) == num_splits_total + len(train), but got: len(id_name_counts)={len(id_name_counts)}, num_splits_total={num_splits_total}, len(train)={len(train)}"
    
    # Write most common program names to disk
    counter = Counter(train["program_name"])
    # TODO: make this a parameter?
    out_file = "data/most_common_program_names/train.txt"
    if not os.path.exists("data/most_common_program_names"):
        os.makedirs("data/most_common_program_names")
    for user_id, count in counter.most_common(1000): 
        with open(out_file, "a") as f:
            f.write(f"{user_id}\t{count}\n")
    print(f"Wrote most common program names in train to {out_file}")

    print("\nCreating lists of tuples of (user_id, program_name) for each split...")

    # create a list of tuples (user_id, program_name) for each split
    train_tuples = list(zip(train["user_id"], train["program_name"]))
    unseen_user_seen_program_tuples = list(
        zip(unseen_user_seen_program["user_id"], unseen_user_seen_program["program_name"])
    )
    unseen_user_unseen_program_tuples = list(
        zip(unseen_user_unseen_program["user_id"], unseen_user_unseen_program["program_name"])
    )
    seen_user_seen_program_tuples = list(
        zip(seen_user_seen_program["user_id"], seen_user_seen_program["program_name"])
    )
    seen_user_unseen_program_tuples = list(
        zip(seen_user_unseen_program["user_id"], seen_user_unseen_program["program_name"])
    )
    
    tuples_by_split = {
        "train": train_tuples,
        "unseen_user_seen_program": unseen_user_seen_program_tuples,
        "unseen_user_unseen_program": unseen_user_unseen_program_tuples,
        "seen_user_seen_program": seen_user_seen_program_tuples,
        "seen_user_unseen_program": seen_user_unseen_program_tuples,
    }
    
    subsets_by_split = {
        "train": train,
        "unseen_user_seen_program": unseen_user_seen_program,
        "unseen_user_unseen_program": unseen_user_unseen_program,
        "seen_user_seen_program": seen_user_seen_program,
        "seen_user_unseen_program": seen_user_unseen_program,
    }

    print_splits(id_name_counts, subsets_by_split)
    
    # SANITY CHECKS
    # for all pairs of tuple lists, make sure there are no overlapping pairs
    for (name1, tuples1) in tuples_by_split.items():
        for (name2, tuples2) in tuples_by_split.items():
            if name1 == name2:
                continue
            assert (
                len(set(tuples1).intersection(set(tuples2))) == 0
            ), f"{name1} and {name2} have {len(set(tuples1).intersection(set(tuples2)))} overlapping tuples"
    
    # make sure all users in seen_user_seen_program are in seen_users
    assert len(set(seen_user_seen_program["user_id"]).difference(set(seen_users))) == 0
    # make sure all programs in seen_user_seen_program are in seen_programs
    assert len(set(seen_user_seen_program["program_name"]).difference(set(seen_programs))) == 0
    
    # make sure all users in seen_user_unseen_program are in seen_users
    assert len(set(seen_user_unseen_program["user_id"]).difference(set(seen_users))) == 0
    # make sure all programs in seen_user_unseen_program are not in seen_programs
    assert len(set(seen_user_unseen_program["program_name"]).intersection(set(seen_programs))) == 0
    
    # make sure all users in unseen_user_seen_program are not in seen_users
    assert len(set(unseen_user_seen_program["user_id"]).intersection(set(seen_users))) == 0
    # make sure all programs in unseen_user_seen_program are in seen_programs
    assert len(set(unseen_user_seen_program["program_name"]).difference(set(seen_programs))) == 0
    
    # make sure all users in unseen_user_unseen_program are not in seen_users
    assert len(set(unseen_user_unseen_program["user_id"]).intersection(set(seen_users))) == 0
    # make sure all programs in unseen_user_unseen_program are not in seen_programs
    assert len(set(unseen_user_unseen_program["program_name"]).intersection(set(seen_programs))) == 0 
    
    return tuples_by_split, seen_users, seen_programs


def print_splits(
    all, subsets_by_split, most_common_k=5
):
    print("SPLITS: # user-program pairs")
    for stub, subset in subsets_by_split.items():
        print(f"- {stub}: {len(subset),} ({len(subset)/len(all)*100:.2f}%)")

    print()
    print("SPLITS: # total programs")
    for stub, subset in subsets_by_split.items():
        print(f'- {stub}: {subset["count"].sum():,} ({subset["count"].sum()/all["count"].sum()*100:.2f}%)')

    print()
    for stub, subset in subsets_by_split.items():
        print()
        print(stub.upper())
        print('# unique user_ids:', len(subset["user_id"].unique()))
       
        # TODO: is this right? 
        num_unique_users = len(subset["user_id"].unique())
        num_programs = subset.shape[0]
        print(f'avg # programs per user: {num_programs/num_unique_users:.2f}')
        
        print(f'# unique program_names: {len(subset["program_name"].unique())}')

        print("most common user_ids:")
        counter = Counter(subset["user_id"])
        for user_id, count in counter.most_common(most_common_k):
            print(f"- {user_id}\t{count:,}")
            
        print("most common program_names:")
        counter = Counter(subset["program_name"])
        for program_name, count in counter.most_common(most_common_k):
            print(f"- {program_name}\t{count:,}")
            
        
def write_splits(
    out_dir="splits",
    test_frac=0.2,
    frac_user_ids_to_holdout=0.01,
    frac_program_names_to_holdout=0.01,
    seed=0,
    id_name_counts_file="data/id_name_counts.txt",
):
    """Writes train, test, heldout_user_test, and heldout_program_test splits to disk.
    - Calls create_train_test_splits to create the splits
    """
    id_name_counts = pd.read_csv(
        id_name_counts_file,
        sep="\t",
        header=None,
        names=["user_id", "program_name", "count"],
    )

    splits_to_tuples, seen_users, seen_programs = create_train_test_splits( 
        id_name_counts,
        test_frac=test_frac,
        frac_user_ids_to_holdout=frac_user_ids_to_holdout,
        frac_program_names_to_holdout=frac_program_names_to_holdout,
        seed=seed,
    )

    print("\nWriting splits to disk...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for stub, tuples in splits_to_tuples.items(): 
        out_file = f"{out_dir}/{stub}.txt"
        print(f"- Writing {stub} to {out_file}:\t({len(tuples)} tuples)")
        with open(out_file, "w") as f:
            for user_id, program_name in tuples:
                f.write(f"{user_id}\t{program_name}\n")
                
    # also write seen_users and seen_programs to disk
    for (name, items) in [("seen_users", seen_users), ("seen_programs", seen_programs)]:
        out_file = f"{out_dir}/{name}.txt"
        print(f"- Writing {name} to {out_file}")
        with open(out_file, "w") as f:
            for item in items:
                f.write(f"{item}\n")


def read_splits(split_dir="splits"):
    print(f"Reading splits from {split_dir}...")
    splits = {}
    files = os.listdir(split_dir)
    files.sort()
    for file in files:
        # TODO: this is hacky, but skip 'username_mappings.db'
        if file == "username_mappings.db":
            continue
        print(f"Reading {file}")
        with open(f"{split_dir}/{file}", "r") as f:
            tuples = [tuple(line.strip().split("\t")) for line in f]
            splits[file.replace(".txt", "")] = set(tuples)
    return splits

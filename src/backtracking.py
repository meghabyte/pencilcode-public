import difflib
import numpy as np
from sklearn.metrics import r2_score
import os

def calculate_edit_distances(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]
    last_line = lines[-1]
    edit_distances = [get_edit_distance(line, last_line) for line in lines]

    return edit_distances

def get_edit_distance(s1, s2):
    seq = difflib.SequenceMatcher(None, s1, s2)
    ratio = seq.ratio()
    edit_distance = int(max(len(s1), len(s2)) * (1 - ratio))
    return edit_distance

def calculate_r2(distances):
    x = np.arange(len(distances))
    y = np.array(distances)

    coefficients = np.polyfit(x, y, 1)
    y_pred = np.polyval(coefficients, x)

    r2 = r2_score(y, y_pred)
    return r2

def calculate_increase_ratio(distances):
    increase_count = sum(1 for i in range(1, len(distances)) if distances[i] > distances[i - 1])
    total_elements = len(distances)
    return increase_count / total_elements

def process_file(file_path):
    edit_distances = calculate_edit_distances(file_path)
    monotonicity = calculate_increase_ratio(edit_distances)
    if(monotonicity == 0):
        return
    print(file_path)
    print(f"Edit Distances: {edit_distances}")
    print(f"Backtracking: {monotonicity}")
    print("\n")


for fd in os.listdir("trace_logs_std/0/00a/"):
    for fn in os.listdir("trace_logs_std/0/00a/"+fd):
        for fs in os.listdir("trace_logs_std/0/00a/"+fd+"/"+fn):
            file_path = "trace_logs_std/0/00a/"+fd+"/"+fn+"/"+fs
            try:
                process_file(file_path)
            except:
                continue

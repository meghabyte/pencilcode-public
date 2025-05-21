import re
import numpy as np
import difflib
from datetime import datetime
from tqdm import tqdm
import pickle

from src.utils import *
from src.nlp_utils import *
from src.execute import CodeExecutor

import multiprocessing
from whoosh import index

from langdetect import detect, detect_langs
from collections import defaultdict

try:
    from src.data_index import *
except:
    pass
    

class Trace:
    def __init__(self, trace_text, username=None, title=None, do_print=False, asset_dir='assets', do_cache_edit_dict=False):
        
        self.do_print = do_print
        self.trace_text = None
        self.title = title
        # same as programs but without "<|endoftext|>"
        self.matched_programs = None
        self.unmatched_programs = None
        self.matched_program_nums = None
        self.unmatched_program_nums = None
        self.matched_dates = None
        self.unmatched_dates = None
        self.eos_reached = False
        self.properties = {} #properties to describe the trace to aid analysis
        self.do_cache_edit_dict = do_cache_edit_dict
        self.edit_dict = None
        self.asset_dir = asset_dir
        with open(f"{self.asset_dir}/colors.txt", 'r') as file:
            program_colors = [line.strip().lower() for line in file]
            self.program_colors = program_colors
            
        try:
            self.index = load_index()
        except:
            self.index = None
         
        
        self.username = username
        
        self.raw_trace_text = trace_text
        # use trace_text, not self.trace_text, bc need to match <|endoftext|>
        found_trace = self.parse_trace(trace_text)
        self.found_trace = found_trace
        
        # cleaned, doesn't include title for now
        self.trace_text = self.get_cleaned_trace_text()
   
    def add_property(self,property_key, property_value):
        self.properties[property_key] = property_value
        
    def get_cleaned_trace_text(self, max_num_programs=None):
        # join matched_programs, matched_dates, formatted as original
        # format as CODE 1 ({date}):\n{program}\nCODE 2 ({date}):\n{program}\n...
        cleaned_trace_text = ""
        assert len(self.matched_program_nums) == len(self.matched_programs) == len(self.matched_dates), f"Lengths of matched_program_nums, matched_programs, and matched_dates do not match: {len(self.matched_program_nums)}, {len(self.matched_programs)}, {len(self.matched_dates)}"
        for idx, (num, prog, date) in enumerate(zip(self.matched_program_nums, self.matched_programs, self.matched_dates)):
            if max_num_programs is not None and idx + 1 > max_num_programs:
                break
            cleaned_trace_text += f"CODE {num} ({date}):\n{prog}"
            # if idx != len(self.matched_program_nums)-1:
            #     cleaned_trace_text += "\n"
            
            # do this for all but the last program bc for last program, will have extra \n?
            # if idx != len(self.matched_program_nums)-1:
            # make sure cleaned trace text is in raw trace text, otherwise cleaning is not working properly
            if cleaned_trace_text not in self.raw_trace_text:
                print(f'Cleaned trace text {cleaned_trace_text} not in raw trace text {self.raw_trace_text}')
                breakpoint()
        cleaned_trace_text = cleaned_trace_text.strip()
        if cleaned_trace_text not in self.raw_trace_text:
            print(f'Cleaned trace text {cleaned_trace_text} not in raw trace text {self.raw_trace_text}')
            #breakpoint()
        assert cleaned_trace_text in self.raw_trace_text, f"Cleaned trace text {cleaned_trace_text} not in raw trace text {self.raw_trace_text}"
        return cleaned_trace_text
        
    def _parse_codes(self, code_list):
        """
        - First get all matches of program number, date, and program text (should filter out cases where program text is empty?)
        - If the last program doesn't end with <|endoftext|>, exclude it
        - If left with no programs, set self.matched_programs = [] and self.matched_program_nums = [] and self.matched_dates = [] (also default to this if no programs match) -> TODO: should self.matched_programs = [""] be the default?
        """
        
        pattern = re.compile(r"CODE (\d+) \((.*?)\):\n(.*)", re.DOTALL)
        # last_pattern is same but with <|endoftext|> at end
        last_pattern = re.compile(r"CODE (\d+) \((.*?)\):\n(.*?)(?=<\|endoftext\|>)", re.DOTALL)

        matched_program_nums, matched_dates, matched_programs = [], [], []
        unmatched_codes = []
        for code_idx, entry in enumerate(code_list):
            if code_idx == len(code_list) - 1:
                match = last_pattern.match(entry)
                if("<|endoftext|>" in entry):
                    self.eos_reached = True
            else:
                match = pattern.match(entry)
            
            if match:
                program_num = int(match.group(1))
                date = match.group(2)
                program = match.group(3)
                
                matched_program_nums.append(program_num)
                matched_dates.append(date)
                matched_programs.append(program)
                
            else:
                unmatched_codes.append(entry)
                
        self.matched_program_nums = matched_program_nums
        self.matched_dates = matched_dates
        self.matched_programs = matched_programs
        
        self.unmatched_codes = unmatched_codes
        self.unmatched_programs = unmatched_codes
        
        # if last program
        
        # make sure the last program is not empty
        if len(self.matched_programs) > 0:
            if self.matched_programs[-1].strip() == '':
                print(f'Last program is empty for trace: {self.raw_trace_text}')
                # TODO: default behavior is to remove the empty program
                # breakpoint()
                # remove last program
                self.matched_programs = self.matched_programs[:-1]
                self.matched_program_nums = self.matched_program_nums[:-1]
                self.matched_dates = self.matched_dates[:-1]
                # need to make sure after removing last program, there is at least one program
                assert len(self.matched_programs) > 0, f'Expected at least one program, but got {len(self.matched_programs)} for trace: {self.raw_trace_text}'
                assert self.matched_programs[-1].strip() != '', f'Last program is empty for trace: {self.raw_trace_text}'
            # assert self.matched_programs[-1].strip() != '', f'Last program is empty for trace: {self.raw_trace_text}'
            
        if self.do_print:
            print(self.raw_trace_text)
            print(f'Matched {len(matched_program_nums)} programs')
            print(f'Unmatched {len(unmatched_codes)} programs')
            for unmatched_code in unmatched_codes:
                print(f'Unmatched program:')
                print(unmatched_code)
                print('-'*100)
               
        if len(unmatched_codes) > 1:
            print(f'Expected at most one unmatched program, but got {len(unmatched_codes)} for trace: {self.raw_trace_text}')
            breakpoint()
        assert len(unmatched_codes) <= 1, f'Expected at most one unmatched program, but got {len(unmatched_codes)} for trace: {self.raw_trace_text}'
            
        if(len(matched_program_nums) == 0):
            if self.do_print:
                print("No Trace Found: ", self.raw_trace_text)
            return False
        return True
    
    def parse_trace(self, trace_text):
        pattern = r"CODE \d+.*?(?=CODE \d+|\Z)"  # Matches each CODE block until the next CODE or end of text
        matches = re.findall(pattern, trace_text, re.DOTALL)
        return self._parse_codes(matches)
    
    def get_last_program_id(self):
        return -1
    
    def get_last_program(self):
        return self.matched_programs[self.get_last_program_id()]
    
    # string representation
    def __repr__(self):
        return self.raw_trace_text
    
    def dump_trace(self,filename):
        print(f'Saving Trace to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    def dump_trace_json(self,filename):
        print(f'Saving Trace to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
    def get_title(self):
        # TODO: a little hacky, but sometimes program_text includes title; if not, title is separate and must be passed in
        if self.title is not None:
            return self.title
        elif self.raw_trace_text is not None:
            return self.raw_trace_text.split('<mask>')[0].strip()
        else:
            return None
        
    def get_diff_dists(self, diff_indices=None):
        if diff_indices is None:
            diff_indices = range(len(self.matched_programs)-1)
            
        if(len(self.matched_programs) <= 1):
            return []
        else:
            diffs = []
            for pi in diff_indices:
                diffs.append(len(self.matched_programs[pi+1].splitlines())-len(self.matched_programs[pi].splitlines()))
            return diffs
        
    
    def get_mean_sequential_edit_distance(self, do_include_boundary=True):
        if len(self.matched_programs) == 1:
            return 0
        elif len(self.matched_programs) == 0:
            boundary_val = 0 if do_include_boundary else np.nan
            return boundary_val
        
        sequential_edit_distances = self.get_sequential_edit_distance()
       
        assert len(sequential_edit_distances) > 0, f"Error: no sequential edit distances found for trace {self.raw_trace_text}"
            
        mean_sequential_edit_distance = np.mean(sequential_edit_distances)
         
        if do_include_boundary:
            assert not np.isnan(mean_sequential_edit_distance), f"Error: mean sequential edit distance is nan for trace {self.raw_trace_text} with sequential edit distances {sequential_edit_distances}"
        return mean_sequential_edit_distance
    
    
    def get_num_occurrences(self, program_text, text_to_find='box'):
        # regex match text_to_find + space or punctuation
        # TODO: not sure if robust
        return len(re.findall(rf'\b{text_to_find}\b', program_text))
    
    def get_num_functions(self, program_text):
        # match all occurrences of '= ->' with arbitrary number of spaces between
        return len(re.findall(r'=\s*->', program_text))
        
    def get_mean_time_diff(self, do_include_boundary=True):
        dates = self.get_dates()
        # TODO: when one date, treat as no time difference
        if len(dates) == 1:
            return 0
        elif len(dates) == 0:
            boundary_val = 0 if do_include_boundary else np.nan
            return boundary_val 
        # TODO: not consistent with get_total_time bc that will default to 0 if only one program but this will default to nan bc taking mean of empty list
        time_diffs = [self.get_time_diff(dates[i], dates[i+1]) for i in range(len(dates)-1)]
        if not do_include_boundary:
            mean = np.nanmean(time_diffs)
        else:
            mean = np.mean(time_diffs)
        if np.isnan(mean):
            print(f"Warning: mean time diff is nan for dates {dates}")
        return mean
    
    def get_total_time(self, do_include_boundary=True):
        dates = self.get_dates()
        last_time = dates[self.get_last_program_id()] 
        first_time = dates[0]
        return self.get_time_diff(first_time, last_time) 
    
    def get_time_diff(self, first_time, last_time, unit='seconds'):
        last_time = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
        first_time = datetime.strptime(first_time, '%Y-%m-%d %H:%M:%S')
        time_diff = last_time - first_time
        time_diff_seconds =  time_diff.total_seconds()
        if unit == 'seconds':
            return time_diff_seconds
        elif unit == 'minutes':
            return time_diff_seconds / 60
        elif unit == 'hours':
            return time_diff_seconds / 3600
        else:
            raise ValueError(f"Invalid unit: {unit}")
                
    def get_programs(self):
        return self.matched_programs
    
    def get_first_timestamp(self):
        return self.matched_dates[0]
    
    def get_last_timestamp(self):
        return self.matched_dates[self.get_last_program_id()]
    
    def get_total_num_comments(self):
        return sum([self.get_num_comments(p) for p in self.matched_programs])
    
    def get_mean_num_comments(self):
        return np.mean([self.get_num_comments(p) for p in self.matched_programs])
 
    def get_num_comments(self, program_text):
        return len([t for t in program_text.split() if t.strip()[0] == "#"])
    
    def get_dates(self):
        return self.matched_dates
    
    def get_len(self):
        return len(self.matched_programs)
    
    def get_creativity(self, creativity_index):
        if(len(self.matched_programs) == 0):
            return None
        else:
            last_program = self.matched_programs[-1]
            s = " ".join(last_program.replace("\n"," ").split()[:creativity_index])
            look_up(self.index, f'"{s}"')
    
    def get_program_number(self, program_index):
        return self.matched_program_nums[program_index]
    
    def get_langs(self, program_index=-1):
        program = self.matched_programs[program_index]
        found_langs = defaultdict(float)
        try:
            for l in  detect_langs(program.lower()):
                found_langs[l.lang] = l.prob
        except:
            return {"en":1}
        return found_langs
         
    def get_colors(self, program_index=-1):
        program = self.matched_programs[program_index]
        found_colors = defaultdict(int)
        for color in self.program_colors:
            found_colors[color] = program.lower().count(color)
        return found_colors #TODO: KL

    def get_year(self, program_index=-1, do_include_boundary=True):
    # TODO: do we need do_include_boundary here?
        # TODO: should not return an error (only would do so if year is malformed?)
        return int(self.matched_dates[program_index][:4])
   
    
    def _get_edit_distance(self, s1, s2):
        seq = difflib.SequenceMatcher(None, s1, s2)
        ratio = seq.ratio()
        edit_distance = int(max(len(s1), len(s2)) * (1 - ratio))
        return edit_distance

    def _calculate_increase_ratio(self, distances):
        increase_count = sum(
            1 for i in range(1, len(distances)) if distances[i] > distances[i - 1]
        )
        total_elements = len(distances)
        return increase_count / total_elements

    def get_programs_upto_last(self):
        """Get all programs up to and including the last program."""
        return self.matched_programs

    def get_backtracking_ratio(self, do_return_dists=False, do_include_boundary=True):
        # if there is one program, treat as backtracking = 0
        if len(self.get_programs_upto_last()) == 1:
            val = 0
            # TODO: for now, this is setting other things like mean_edit_distance_from_end to 0; is this what we want?
            if do_return_dists:
                return (val, val, val, [val])
            return (val, val, val)
        if len(self.get_programs_upto_last()) == 0:
            # treat as nan or not depending on whether including boundary
            boundary_val = 0 if do_include_boundary else np.nan
            if do_return_dists:
                return (boundary_val, boundary_val, boundary_val, [boundary_val])
            return (boundary_val, boundary_val, boundary_val)
        dists = [self._get_edit_distance(t, self.get_last_program()) for t in self.matched_programs[:self.get_last_program_id()]]
        # print(dists)
        if do_return_dists:
            return self._calculate_increase_ratio(dists), sum(dists), np.mean(dists), dists
        return self._calculate_increase_ratio(dists), sum(dists), np.mean(dists)
    
    # def get_backtracking_diff_dists(self):
    #     dists = [self._get_edit_distance(t, self.get_last_program()) for t in self.matched_programs[:self.get_last_program_id()]]
    #     diff_dists = [0]
    #     for di in range(1, len(dists)):
    #         diff_dists.append(dists[di]-dists[di-1])
    #     return diff_dists
    def get_backtracking_diff_dists(self, diff_indices=None):
        if diff_indices is None:
            # until but not including last program
            diff_indices = range(len(self.matched_programs)-1)
        # get distances for all programs
        dists = [self._get_edit_distance(t, self.get_last_program()) for t in self.matched_programs]
        diff_dists = []
        for di in diff_indices:
            diff_dist = dists[di+1] - dists[di]
            diff_dists.append(diff_dist)
        assert len(diff_dists) == len(diff_indices), f'Expected len(diff_dists) = len(diff_indices) - 1, but got {len(diff_dists)} != {len(diff_indices)-1}'
        return diff_dists
        
    
    def get_max_backtracking_index(self, do_return_dists=True):
        dists = [self._get_edit_distance(t, self.get_last_program()) for t in self.matched_programs[:self.get_last_program_id()]]
        diff_dists = [0]
        for di in range(1, len(dists)):
            diff_dists.append(dists[di]-dists[di-1])
        max_index = np.argmax(diff_dists)
        if(do_return_dists):
            return max_index, dists
        else:
            return max_index
        
    def get_min_backtracking_index(self, do_return_dists=True):
        dists = [self._get_edit_distance(t, self.get_last_program()) for t in self.matched_programs[:self.get_last_program_id()]]
        diff_dists = [0]
        for di in range(1, len(dists)):
            diff_dists.append(dists[di]-dists[di-1])
        min_index = np.argmin(diff_dists)
        if(do_return_dists):
            return min_index, dists
        else:
            return min_index
    
    def reference_bleu(self, ref_t, ref_idx=-1, self_idx=-1, mode="last"):
        if mode == "last":
            texts = [self.matched_programs[self_idx]]
            refs = [ref_t.matched_programs[ref_idx]]
        elif mode == "all":
            texts = [self.trace_text]
            refs = [ref_t.trace_text]
        else:
            raise ValueError("mode should be either 'last' or 'all'")
        return quality_bleu(texts=texts, refs=refs)
        
    
    def get_sequential_edit_distance(self, diff_indices=None):
        """Get edit distance between each pair of consecutive programs."""
        if diff_indices is None:
            diff_indices = range(len(self.matched_programs)-1)
        dists = []
        for i in diff_indices:
            # if i == -1, then get edit distance from empty program to first program
            if i == -1:
                dists.append(self._get_edit_distance("", self.matched_programs[i+1]))
            else:
                dists.append(self._get_edit_distance(self.matched_programs[i], self.matched_programs[i+1]))
        return dists
    
    def is_degenerate(self):
        is_degenerate = (len(self.matched_program_nums) == 0)
        return is_degenerate
    
    def reached_eos(self):
        return self.eos_reached
    
    def get_num_unmatched(self):
        return len(self.unmatched_codes)
    
    def get_last_program(self):
        return self.matched_programs[-1]
    
    
    def get_metrics(self, do_include_boundary=True):
        last_metrics = self.get_metrics_on_program(program_idx=self.get_last_program_id(), do_include_boundary=do_include_boundary)
        all_metrics = self.get_metrics_on_all(do_include_boundary=do_include_boundary)
        
        
        r = {f'{k}_last': v for k, v in last_metrics.items()}
        r.update({f'{k}_all': v for k, v in all_metrics.items()})
        
        # add whether to include boundary cases in metric name
        new_r = {f'{k}_boundary={do_include_boundary}': v for k, v in r.items()}
        return new_r
    
    def get_metrics_on_program(self, program_idx=-1, do_include_boundary=True):
        # all the metrics on particular program
        metrics = {
            "trace_year": int(self.get_year(program_idx, do_include_boundary=do_include_boundary)),
            "program_number": int(self.get_program_number(program_idx)),
            "num_occurrences_box": int(self.get_num_occurrences(self.matched_programs[program_idx], text_to_find='box')),
            "num_occurrences_await": int(self.get_num_occurrences(self.matched_programs[program_idx], text_to_find='await')),
            "num_occurrences_turtle": int(self.get_num_occurrences(self.matched_programs[program_idx], text_to_find='turtle')),
            "num_functions": int(self.get_num_functions(self.matched_programs[program_idx])),
            "num_comments": int(self.get_num_comments(self.matched_programs[program_idx])),
        }
        return metrics
        
    def get_metrics_on_all(self, do_include_boundary=True):
        # metrics on all programs
        backtracking_ratio, _, mean_edit_distance_from_end, edit_distances_from_last = self.get_backtracking_ratio(do_return_dists=True, do_include_boundary=do_include_boundary)
        # sequential_edit_distances = self.get_sequential_edit_distance()
        
        # TODO: is this the right default behavior when there is only one program?
        # boundary_val = 0 if do_include_boundary else np.nan
        # mean_sequential_edit_distance = np.mean(sequential_edit_distances) if len(sequential_edit_distances) >= 1 else boundary_val
        
        
        
        mean_sequential_edit_distance = self.get_mean_sequential_edit_distance(do_include_boundary=do_include_boundary)
        
        num_occurrences_box = [self.get_num_occurrences(p, text_to_find='box') for p in self.matched_programs]
        num_occurrences_turtle = [self.get_num_occurrences(p, text_to_find='turtle') for p in self.matched_programs]
        num_functions = [self.get_num_functions(p) for p in self.matched_programs]
        all_metrics = {
            "trace_length": int(self.get_len()),
            "backtracking_ratio": float(backtracking_ratio),
            "mean_edit_distance_from_end": float(mean_edit_distance_from_end),
            "first_to_last_edit_distance": float(edit_distances_from_last[0]), 
            "mean_sequential_edit_distance": float(mean_sequential_edit_distance), 
            "mean_time_diff": float(self.get_mean_time_diff(do_include_boundary=do_include_boundary)),
            "total_time": float(self.get_total_time(do_include_boundary=do_include_boundary)),
            "sum_num_occurrences_box": int(sum(num_occurrences_box)),
            "sum_num_occurrences_turtle": int(sum(num_occurrences_turtle)),
            "sum_num_functions": int(sum(num_functions)),
            "mean_num_occurrences_box": float(np.mean(num_occurrences_box)),
            "mean_num_occurrences_turtle": float(np.mean(num_occurrences_turtle)),
            "mean_num_functions": float(np.mean(num_functions)),
        }
        
        # gives sum of number of occurrences of each edit type
        # edit_dict = self.get_edit_dict()
        # # for each edit, add sum and mean
        # for edit_type, count in edit_dict.items():
        #     all_metrics[f'sum_num_occurrences_{edit_type}'] = int(count)
        #     all_metrics[f'mean_num_occurrences_{edit_type}'] = count/len(self.matched_programs)
        
        edit_metrics = self.get_edit_metrics(do_print=False)
        for k, v in edit_metrics.items():
            all_metrics[k] = v
        
        # if len(sequential_edit_distances) < 1:
        #     if all_metrics['trace_length'] != 1:
        #         breakpoint()
        #     assert all_metrics['trace_length'] == 1, f"Trace length {all_metrics['trace_length']} does not match sequential edit distances {sequential_edit_distances}: {self.trace_text}"
        
        return all_metrics
    
    def get_edit_metrics(self, do_print=False, diff_indices=None):
        edit_metrics = self.get_clean_edit_dict(diff_indices=diff_indices, do_print=do_print)
        temp_r = {
            "lines_added": edit_metrics['lines_added'],
            "lines_deleted": edit_metrics['lines_deleted'],
            "lines_modified": edit_metrics['lines_modified'],
            "color_changes": edit_metrics['color_changes'],
            "number_changes": edit_metrics['number_changes'],
            "comment_additions": edit_metrics['comment_additions'],
            "function_additions": edit_metrics['function_additions'],
            
            # add these bc if diff_indices is more than 1 element, want to get numbers across trace
            'num_large_add': edit_metrics['edit_categories']['large_add'],
            'num_large_del': edit_metrics['edit_categories']['large_del'],
            'num_small_add': edit_metrics['edit_categories']['small_add'],
            'num_small_del': edit_metrics['edit_categories']['small_del'],
            'num_modify_existing': edit_metrics['edit_categories']['modify_existing'],
        }
        r = {f'edit_sum_{edit_type}': temp_r[edit_type] for edit_type in temp_r}    
        r.update({f'edit_mean_{edit_type}': temp_r[edit_type]/len(self.matched_programs) for edit_type in temp_r})
        return r

    def get_clean_edit_dict(self, diff_indices=None, do_print=False, collect_examples=False):
        """
        Returns a cleaner version of edit tracking with the following categories:
        1. Mutually exclusive edit types (one-hot encoded):
           - large_add: Addition of 5+ lines
           - large_del: Deletion of 5+ lines
           - small_add: Addition of 1-4 lines
           - small_del: Deletion of 1-4 lines
           - modify_existing: Modification of existing lines
        2. Visual property changes (counts):
           - color_changes: Number of color-only changes
           - number_changes: Number of number-only changes
        3. Code structure changes (counts):
           - comment_additions: Number of comment lines added
           - function_additions: Number of function definitions added
        4. Line change counts:
           - lines_added: Total number of lines added
           - lines_deleted: Total number of lines deleted
           - lines_modified: Total number of lines modified (any type of change)
        """
        
        if collect_examples:
            raise NotImplementedError("collect_examples not implemented")
        
        # only cache for diff_indices = None
        if diff_indices is None and self.do_cache_edit_dict and self.edit_dict is not None:
            return self.edit_dict
        
        if diff_indices is None:
            diff_indices = range(len(self.matched_programs)-1)
            
        if do_print:
            print("\n=== Starting Edit Analysis ===")
            print(f"Analyzing {len(diff_indices)} program pairs")
            
        # Initialize results dictionary
        edit_categories = {
            'large_add': 0,
            'large_del': 0,
            'small_add': 0,
            'small_del': 0,
            'modify_existing': 0,
        }
        results = {
            # Mutually exclusive edit types (one-hot encoded)
            'edit_categories': edit_categories,
            
            # Visual property changes (counts)
            'color_changes': 0,
            'number_changes': 0,
            
            # Code structure changes (counts)
            'comment_additions': 0,
            'function_additions': 0,
            
            # Line change counts
            'lines_added': 0,
            'lines_deleted': 0,
            'lines_modified': 0,  # New counter for modified lines
        }
        
        # Process each pair of consecutive programs
        for pi in diff_indices:
            if do_print:
                print(f"\n--- Analyzing Programs {pi} -> {pi+1} ---")
                
            # Get differences between programs using difflib
            # if pi == -1, then use program1 = "", ie get diff from empty program to first program
            if pi == -1:
                program1 = "".splitlines()
                program2 = self.matched_programs[pi+1].splitlines()
            else:
                program1 = self.matched_programs[pi].splitlines()
                program2 = self.matched_programs[pi+1].splitlines()
            
            if do_print:
                print(f"Program {pi} has {len(program1)} lines")
                print(f"Program {pi+1} has {len(program2)} lines")
            
            # Use difflib to get differences
            differ = difflib.Differ()
            diffs = list(differ.compare(program1, program2))
            
            if do_print:
                print("\nDiff output:")
                print('\n'.join(diffs))
            
            # First pass: identify all changes
            token_changes = []
            processed_lines = set()  # Track which lines we've already processed
            lines_modified = 0  # Counter for this program pair
            modified_lines = []  # Track which lines were modified and how
            
            for i, diff in enumerate(diffs):
                if diff.startswith('  '):  # Skip unchanged lines
                    continue
                    
                if diff.startswith('? '):  # Skip the ? lines that show where changes occurred
                    continue
                    
                if diff.startswith('+ ') or diff.startswith('- '):
                    line = diff[2:] if len(diff) > 2 else ''
                    
                    # Skip if we've already processed this line
                    if line in processed_lines:
                        continue
                    
                    # Find the corresponding line in the other program
                    other_line = ''
                    if diff.startswith('+ '):
                        # Look for the corresponding line in program1
                        for j in range(i-1, -1, -1):
                            if diffs[j].startswith('- '):
                                other_line = diffs[j][2:]
                                processed_lines.add(other_line)  # Mark both lines as processed
                                processed_lines.add(line)
                                break
                    else:  # diff.startswith('- ')
                        # Look for the corresponding line in program2
                        for j in range(i+1, len(diffs)):
                            if diffs[j].startswith('+ '):
                                other_line = diffs[j][2:]
                                processed_lines.add(other_line)  # Mark both lines as processed
                                processed_lines.add(line)
                                break
                    
                    if do_print:
                        print(f"\nAnalyzing lines:")
                        print(f"  Line 1: {line}")
                        print(f"  Line 2: {other_line}")
                    
                    # If we found a corresponding line, this is a modification
                    if other_line:
                        lines_modified += 1
                        modified_lines.append((line, other_line))
                        
                        # Check for specific types of changes
                        if differs_only_by_color(line, other_line, asset_dir=self.asset_dir):
                            token_changes.append(('color', line, other_line))
                            if do_print:
                                print("  Found: COLOR CHANGE")
                        elif differs_only_by_number(line, other_line):
                            token_changes.append(('number', line, other_line))
                            if do_print:
                                print("  Found: NUMBER CHANGE")
                        else:
                            if do_print:
                                print("  Found: OTHER MODIFICATION")
            
            # Second pass: count actual line changes (excluding modified lines)
            lines_added = 0
            lines_deleted = 0
            
            for diff in diffs:
                if diff.startswith('  '):  # Skip unchanged lines
                    continue
                    
                if diff.startswith('? '):  # Skip the ? lines that show where changes occurred
                    continue
                    
                if diff.startswith('+ ') or diff.startswith('- '):
                    line = diff[2:] if len(diff) > 2 else ''
                    
                    # Skip if this line is part of a modification
                    if line in processed_lines:
                        continue
                    
                    if diff.startswith('+ '):
                        lines_added += 1
                        # Check for comment/function additions
                        stripped_line = line.strip()
                        if stripped_line and stripped_line[0] == '#':
                            results['comment_additions'] += 1
                            if do_print:
                                print("  Found: COMMENT ADDITION")
                        # Match various function definition styles:
                        function_patterns = [
                            r'\s*->\s*'
                        ]
                        if any(re.search(pat, stripped_line) for pat in function_patterns):
                            results['function_additions'] += 1
                            if do_print:
                                print(f'Line: {stripped_line}')
                                print("  Found: FUNCTION ADDITION")
                    else:  # diff.startswith('- ')
                        lines_deleted += 1
            
            # Update results
            results['lines_added'] += lines_added
            results['lines_deleted'] += lines_deleted
            results['lines_modified'] += lines_modified
            
            # Count token-level changes
            for change_type, _, _ in token_changes:
                if change_type == 'color':
                    results['color_changes'] += 1
                elif change_type == 'number':
                    results['number_changes'] += 1
            
            if do_print:
                print(f"\nChanges found:")
                print(f"  Lines modified: {lines_modified}")
                print(f"  Color changes: {sum(1 for t,_,_ in token_changes if t == 'color')}")
                print(f"  Number changes: {sum(1 for t,_,_ in token_changes if t == 'number')}")
                print(f"  Other modifications: {lines_modified - len(token_changes)}")
            
            # Determine edit type (mutually exclusive)
            if lines_added >= 5:
                results['edit_categories']['large_add'] += 1
                if do_print:
                    print("  Classified as: LARGE ADDITION")
            elif lines_deleted >= 5:
                results['edit_categories']['large_del'] += 1
                if do_print:
                    print("  Classified as: LARGE DELETION")
            elif lines_added > 0:
                results['edit_categories']['small_add'] += 1
                if do_print:
                    print("  Classified as: SMALL ADDITION")
            elif lines_deleted > 0:
                results['edit_categories']['small_del'] += 1
                if do_print:
                    print("  Classified as: SMALL DELETION")
            else:
                assert lines_added == 0 and lines_deleted == 0, f"Expected 0 lines added and deleted, but got {lines_added} and {lines_deleted}"
                results['edit_categories']['modify_existing'] += 1
                if do_print:
                    print("  Classified as: MODIFY EXISTING")
                        
            if do_print:
                print(f"\nSummary for programs {pi} -> {pi+1}:")
                print(f"  Lines added: {lines_added}, Lines deleted: {lines_deleted}, Lines modified: {lines_modified}")
                print(f"  Color changes: {results['color_changes']}, Number changes: {results['number_changes']}")
                print(f"  Comments added: {results['comment_additions']}, Functions added: {results['function_additions']}")
                
        if do_print:
            print("\n=== Final Results ===")
            for key, value in results.items():
                print(f"{key}: {value}")
                
        # make sure exactly one of the edit categories is 1
        assert sum(results['edit_categories'].values()) == len(diff_indices), f"Expected exactly one edit category to be 1 for each program pair, but got {results['edit_categories']} for {len(diff_indices)} program pairs"
                
        if self.do_cache_edit_dict:
            self.edit_dict = results
                
        return results
import re
import numpy as np
import difflib
from datetime import datetime
from tqdm import tqdm
import pickle
from src.metrics import Trace
from src.utils import *
from src.nlp_utils import *
from src.execute import CodeExecutor
import json

import multiprocessing
from whoosh import index
from collections import defaultdict


class TraceCollection:
    """Class to hold a collection of traces. Can use to hold a collection of traces from a single user or multiple users. Used for holding traces for a particular student in Student/StudentCollection class."""
    def __init__(self, traces=None, code_executor=None, do_compute_code_execution=True):
        if traces is None:
            traces = []
        self.traces = traces
        self.code_executor = code_executor
        self.aggregate_metrics = None
        self.metric_to_num_nans = None
        self.do_compute_code_execution = do_compute_code_execution
        
    def dump_trace_collection(self,filename):
        print(f'Saving Trace Collection to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    def dump_trace_collection_json(self,filename):
        print(f'Saving Trace Collection to {filename}')
        trace_texts = [[t.raw_trace_text, t.username, t.title] for t in self.traces]
        with open(filename, 'w') as f:
            json.dump(trace_texts, f, ensure_ascii=False, indent=2)
            
    def load_trace_collection_json(self,filename):
        with open(filename, "r", encoding="utf-8") as f:
            raw_trace_texts = json.load(f)
            for t in raw_trace_texts:
                self.traces.append(Trace(trace_text=t[0], username=t[1], title=t[2]))
            #print(f'Loaded {len(self.traces)} traces')
    
    def create_dictionary(self, raw_trace_text=False):
        tc_dict = defaultdict(list)
        for t in self.traces:
            entry_key = (t.username, t.title)
            if(raw_trace_text):
                tc_dict[entry_key].append(t.raw_trace_text)
            else:
                tc_dict[entry_key].append(t)
        return tc_dict
        
        
    def add_trace(self, trace):
        self.traces.append(trace)
       
    def get_num_degenerate(self):
        return len([t for t in self.traces if t.is_degenerate()])
        
    def get_titles(self):
        return [t.get_title() for t in self.traces]
    
    def creativity(self):
        return [t.get_creativity() for t in self.traces]
   
    def get_model_probs(self, model, tokenizer, user_hash, device):
        # get perplexity for each trace
        scores = []
        for t in self.traces:
            input_ids = tokenizer.encode(t.trace_text, return_tensors='pt').to(device)
            print(f'Input ids: {input_ids.shape}')
            usernames = user_hash.get_hash(t.username)
            print(f'Usernames: {usernames}')
            username_tensor = torch.tensor([usernames]).to(device)
            print(f'Username tensor: {username_tensor}')
            # shift to get the labels
            label_ids = input_ids.clone()
            label_ids[:, 0] = -100
            outputs = model(input_ids, labels=label_ids, username_hashes=username_tensor)
            scores.append(outputs.loss.item())
        return scores
        
    
    def reference_bleu(self, ref_tc, mode="last"):
        if mode == "last":
            texts = [t.get_last_program() for t in self.traces]
            refs = [t.get_last_program() for t in ref_tc.traces]
        elif mode == "all":
            texts = [t.trace_text for t in self.traces]
            refs = [t.trace_text for t in ref_tc.traces]
        else:
            raise ValueError("mode should be either 'last' or 'all'")
        return quality_bleu(texts=texts, refs=refs)
    
    def self_bleu(self):
        return [self_bleu([t.get_last_program() for t in self.traces])]
    
    def self_jaccard(self):
        return [self_jaccard([t.get_last_program() for t in self.traces])]
       
    def get_code_execution_results(self, program_index=None, timeout_seconds_execute=5, timeout_seconds_navigate=5, max_concurrent=15):
        # try:
        if self.code_executor is None:
            self.code_executor = CodeExecutor()
        if (program_index is None):
            last_progs = [t.get_last_program() for t in self.traces]
            success_results, end_reached_results, failed_to_reach_end_results, reached_end_but_had_errors_results = self.code_executor.get_execute_results(last_progs, timeout_seconds_execute=timeout_seconds_execute, timeout_seconds_navigate=timeout_seconds_navigate, max_concurrent=max_concurrent)
        else:
            progs = [t.matched_programs[program_index] for t in self.traces]
            print(f'First 10 chars of each program: {[p[:10] for p in progs]}')
            success_results, end_reached_results, failed_to_reach_end_results, reached_end_but_had_errors_results = self.code_executor.get_execute_results(progs, timeout_seconds_execute=timeout_seconds_execute, timeout_seconds_navigate=timeout_seconds_navigate, max_concurrent=max_concurrent)
        # except Exception as e:
            # print(f"Error during execution: {e}")
        # success_results is list of bools
        success_results = [int(s) for s in success_results]
        # print(success_results)     
        return success_results 
        
    def compute_code_execution_results_batches(self, program_index=None, timeout_seconds_execute=5, timeout_seconds_navigate=5, max_concurrent=15, write_cache_idx=100):
        # try:
        if self.code_executor is None:
            self.code_executor = CodeExecutor()
        if (program_index is None):
            progs = [t.get_last_program() for t in self.traces]
        else:
            progs = [t.matched_programs[program_index] for t in self.traces]
            print(f'First 10 chars of each program: {[p[:10] for p in progs]}')
        
         # pre-compute last program results
        print(f'Computing code execution results for {len(progs)} programs...')
        # iterate in batches
        batch_size = write_cache_idx
        num_batches = len(progs) // batch_size + 1
        pbar = tqdm(range(0, len(progs), batch_size), desc=f"Computing code execution results with batch size={batch_size}", total=num_batches)
        prev_num_cache = len(self.code_executor.cache)
        
        all_success_results = []
        for i in pbar:
            batch = progs[i:i+batch_size]
            success_results, _, _, _ = self.code_executor.get_execute_results(batch, timeout_seconds_execute=timeout_seconds_execute, timeout_seconds_navigate=timeout_seconds_navigate, max_concurrent=max_concurrent)
            # breakpoint() 
            self.code_executor.write_cache(do_print=False)
            pbar.set_description(f"Computing code execution results with batch size={batch_size} (+{len(self.code_executor.cache)-prev_num_cache} cache entries)")
            all_success_results.extend(success_results)

        # success_results, _, _, _ = self.code_executor.get_execute_results(progs, timeout_seconds_execute=timeout_seconds_execute, timeout_seconds_navigate=timeout_seconds_navigate, max_concurrent=max_concurrent)
            # print(f"Error during execution: {e}")
        # success_results is list of bools
        all_success_results = [int(s) for s in all_success_results]
        # print(success_results)     
        return all_success_results
    
    def get_individual_metrics(self, do_include_boundary=True):
        """Get metrics for each trace."""
        # print(f'Getting individual metrics for {len(self.traces)} traces...')
        # TODO: this code can probably be sped up; store as dict of lists?
        all_metrics = []
        # metric_to_vals is a dictionary of metric -> list of values for that metric across all traces
        metric_to_vals = {}
        metric_to_num_nans = {}
        for trace in self.traces:
            m = trace.get_metrics(do_include_boundary=do_include_boundary)
            all_metrics.append(m)
            for k, v in m.items():
                if k not in metric_to_vals:
                    metric_to_vals[k] = []
                metric_to_vals[k].append(v)
                
        for k, v in metric_to_vals.items():
            num_nan = np.sum(np.isnan(v))
            metric_to_num_nans[k] = num_nan
        # print(f'Done.')
      
        if self.do_compute_code_execution: 
            # TODO: add 'last' to metric name bc it's computed on last program and get_metrics() does that for all metrics computed on last program
            success_results = self.get_code_execution_results() 
            metric_key = 'executed_without_error_last'
            # now add back to all_metrics and metric_to_vals
            metric_to_vals[metric_key] = success_results
        
            for i, trace in enumerate(self.traces):
                all_metrics[i][metric_key] = success_results[i]
            
        return all_metrics, metric_to_vals, metric_to_num_nans
    
    def get_aggregate_metrics(self):
        assert self.aggregate_metrics is not None, f"Error: aggregate_metrics has not been set for this TraceCollection"
        return self.aggregate_metrics
    
    def set_aggregate_metrics(self, do_include_boundary=True):
        """Get aggregate metrics for all traces."""
        
        all_metrics, metric_to_vals, metric_to_num_nans = self.get_individual_metrics(do_include_boundary=do_include_boundary)
        # aggregate metrics is a dictionary of metric -> mean across all traces
        aggregate_metrics = {}
        
        # First get metrics that are means of metrics on individual programs
        # take mean of all metrics, except metrics_to_skip
        metrics_to_skip = []
        for key in all_metrics[0].keys():
            if key in metrics_to_skip:
                continue
            
            # if any are nan in metric_to_vals[key], print out
            if any(np.isnan(metric_to_vals[key])) and do_include_boundary:
                print(f"Warning: {key} has nans in metric_to_vals: {metric_to_vals[key]}")
                
            if all(np.isnan(metric_to_vals[key])):
                print(f"Warning: all values for {key} are nan")
                print(f"Values: {metric_to_vals[key]}")
                breakpoint()
                
            # make sure not all are nan
            # TODO: also want to check this if not including boundary?
            # if do_include_boundary:
            assert not all(np.isnan(metric_to_vals[key])), f"Error: all values for {key} are nan"
            
            # nanmean, skip nans
            aggregate_metrics[key] = np.nanmean(metric_to_vals[key])
            
            # Then get metrics that are metrics on all programs, eg num_traces
        # also add other metrics that aren't means across all traces
        aggregate_metrics['num_traces'] = len(all_metrics)
        aggregate_metrics['num_unique_titles'] = len(set([t.get_title() for t in self.traces]))
       
        if self.aggregate_metrics is None:
            self.aggregate_metrics = {} 
        self.aggregate_metrics.update(aggregate_metrics)
        
        self.metric_to_num_nans = metric_to_num_nans
        
        return self.aggregate_metrics

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import torch
import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from src.metrics.trace import Trace
from src.utils import load_model, UsernameHash
import matplotlib.pyplot as plt
from src.eval.probe import LinearProbe
from collections import Counter
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
import scipy.stats as stats
from src.execute import CodeExecutor
from src.utils import load_model, UsernameHash, print_dict, set_seed
from src.generation import rollout
from src.global_vars import *

class BaseEmbeddingExtractor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        data_dir: str = './data',
        batch_size: int = 32
    ):
        """Initialize the base embedding extractor.
        
        Args:
            model_path: Path to the model checkpoint (if not providing model/tokenizer)
            model: Pre-trained model (if not providing model_path)
            tokenizer: Pre-trained tokenizer (if not providing model_path)
            device: Device to run the model on
            data_dir: Directory for data files
            batch_size: Batch size for processing multiple inputs
        """
        if model_path is not None:
            assert model is None and tokenizer is None, "Cannot provide both model_path and model/tokenizer"
            self.model, self.tokenizer = load_model(model_path, device=device)
        else:
            assert model is not None and tokenizer is not None, "Must provide either model_path or both model and tokenizer"
            self.model = model.to(device)
            self.tokenizer = tokenizer
            
        self.device = device
        print(f'Device: {self.device}')
        self.batch_size = batch_size
        self.captured_embeddings = []
        self.user_hash = UsernameHash(data_dir=data_dir)

class StudentEmbeddingExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        mlp_layer: int = 2,
        **kwargs
    ):
        """Initialize the student embedding extractor.
        
        Args:
            model_path: Path to the model checkpoint (if not providing model/tokenizer)
            model: Pre-trained model (if not providing model_path)
            tokenizer: Pre-trained tokenizer (if not providing model_path)
            mlp_layer: Which MLP layer to extract embeddings from (1 or 2)
            **kwargs: Additional arguments passed to BaseEmbeddingExtractor
        """
        super().__init__(model_path=model_path, model=model, tokenizer=tokenizer, **kwargs)
        self.mlp_layer = mlp_layer
        
        # Register the hook on the specified MLP layer
        layer_name = f"student_embedding_model.mlp.{mlp_layer}"
        embedding_layer = dict(self.model.named_modules())[layer_name]
        embedding_layer.register_forward_hook(self._capture_output)
        
    def _capture_output(self, module, input, output):
        """Hook function to capture the output of a layer."""
        self.captured_embeddings.append(output.squeeze().cpu().detach())

    def get_embedding(
        self,
        program_name: str,
        username: str,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = 10,
    ) -> np.ndarray:
        """Get student embedding for a single trace.
        
        Args:
            trace: The code trace to get embedding for
            username: Username associated with the trace
            max_length: Maximum sequence length (optional)
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        self.captured_embeddings = []
        
        rollout(
            self.model,
            self.tokenizer,
            self.user_hash,
            [program_name],
            usernames=[username],
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            device=self.device,
            return_trace=False,
            do_generate_until_eos=False
        )
            
        if not self.captured_embeddings:
            raise RuntimeError("No embeddings were captured")
            
        return self.captured_embeddings[0].numpy()
    
class ProgramEmbeddingExtractor(BaseEmbeddingExtractor):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        layer: int = -1,  # Last layer by default
        aggregation: str = "mean",
        **kwargs
    ):
        """Initialize the program embedding extractor.
        
        Args:
            model_path: Path to the model checkpoint (if not providing model/tokenizer)
            model: Pre-trained model (if not providing model_path)
            tokenizer: Pre-trained tokenizer (if not providing model_path)
            layer: Which transformer layer to extract embeddings from (-1 for last layer)
            aggregation: How to aggregate token embeddings ("mean" or "last")
            **kwargs: Additional arguments passed to BaseEmbeddingExtractor
        """
        super().__init__(model_path=model_path, model=model, tokenizer=tokenizer, **kwargs)
        self.layer = layer
        self.aggregation = aggregation
        assert aggregation in ["mean", "last"], "Aggregation must be 'mean' or 'last'"
        
        # Register the hook on the specified transformer layer
        if layer == -1:
            layer = len(self.model.transformer.h) - 1
        self.layer = layer
        transformer_layer = self.model.transformer.h[layer]
        self.hook_handle = transformer_layer.register_forward_hook(self._capture_output)
    
    def change_layer(self, new_layer):
        self.hook_handle.remove()
        if new_layer == -1:
            new_layer = len(self.model.transformer.h) - 1
        self.layer = new_layer
        transformer_layer = self.model.transformer.h[new_layer]
        self.hook_handle = transformer_layer.register_forward_hook(self._capture_output)
        
    def _capture_output(self, module, input, output):
        """Hook function to capture the output of a transformer layer."""
        # For GPT models, output is typically a tuple where the first element 
        # contains the hidden states
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.captured_embeddings.append(hidden_states.cpu().detach())

    def _aggregate_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """Aggregate token embeddings into a single vector."""
        if self.aggregation == "mean":
            # Average across all tokens
            return embeddings.mean(dim=1).numpy()
        elif self.aggregation == "last":
            # Use the last token's embedding
            return embeddings[:, -1].numpy()
        else:
            raise ValueError(f'Invalid aggregation: {self.aggregation}')

    def get_embedding(
        self,
        code: str,
        program_name: str,
        username: str,
        do_print: bool = False,
        do_return_num_input_tokens: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        """Get student embedding for a single trace.
        
        Args:
            trace: The code trace to get embedding for
            username: Username associated with the trace
            max_length: Maximum sequence length (optional)
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        self.captured_embeddings = []
        
        extra_text_ids = self.tokenizer([code], return_tensors='pt').to(self.device)["input_ids"]
        
        program_tokens = self.tokenizer([program_name], return_tensors='pt', padding='max_length', max_length=PREFIX_MAX_LENGTH, truncation=True).to(self.device)
        program_token_ids = program_tokens['input_ids']
        program_token_ids[program_token_ids == self.tokenizer.pad_token_id] = self.tokenizer.mask_token_id
        start_tokens= self.tokenizer.batch_encode_plus(['<start>'], add_special_tokens=False, return_tensors='pt').to(self.device)
        start_token_ids = start_tokens['input_ids']
        
        input_tensor_ids = torch.cat([program_token_ids,start_token_ids, extra_text_ids],dim=1)
        
        num_input_tokens_before_truncation = input_tensor_ids.shape[1]
        
        # truncate up to 1024 tokens
        # TODO: does this need to be 1023? if 1024, get errors
        input_tensor_ids = input_tensor_ids[:, :1023]
        
        usernames = [self.user_hash.get_hash(username)]
        username_hashes = torch.tensor(usernames).to(self.device)
        
        # print(f'Input tensor ids: {input_tensor_ids.shape}')
        # print(f'Username hashes: {username_hashes.shape}')
        out = self.model(input_tensor_ids, username_hashes=username_hashes)
        
        if do_print:
            # decode input_tensor_ids
            input_text = self.tokenizer.decode(input_tensor_ids[0], skip_special_tokens=False)
            print(f'Username hashes: {username_hashes}')
            print(f'Input text:\n{input_text}')
        
            
        if not self.captured_embeddings:
            raise RuntimeError("No embeddings were captured")
            
        assert len(self.captured_embeddings) == 1, f'Expected 1 embedding, got {len(self.captured_embeddings)}'
        if do_return_num_input_tokens:
            return self._aggregate_embeddings(self.captured_embeddings[0]), num_input_tokens_before_truncation
        else:
            return self._aggregate_embeddings(self.captured_embeddings[0])
    
def extract_embeddings_and_metadata(traces, users, users_to_condition_on, embedding_extractor, condition_code_format='full', do_mask_title=False):
    """Extract embeddings and remaining attempts for a list of traces."""
    
    valid_condition_code_formats = ['full', 'last']
    if condition_code_format not in valid_condition_code_formats:
        raise ValueError(f'Invalid condition code format: {condition_code_format}')
    
    embeddings = []
    start_texts = []
    metadata_list = []
    num_traces_with_break = 0
    # code_executor = CodeExecutor()
    # num_ex_with_break_list = []
    
    keys = set()
    
    num_issues = 0
    
    for trace_idx, (trace, username, user_to_condition_on) in tqdm(enumerate(zip(traces, users, users_to_condition_on)), total=len(traces), desc='Extracting embeddings and targets'):
        num_total_programs = len(trace.matched_programs)
        
        trace_metadata = []
        trace_embeddings = []
        trace_start_texts = []
        num_ex_with_break_per_trace = 0
        
        title = trace.title
        
        for max_num_programs in range(0,num_total_programs+1):
            do_print = trace_idx in [0,1,2,100, 1000]
            
            prog_idx = max_num_programs - 1
            
            key = (username, title, max_num_programs)
            
            is_final_program = max_num_programs == num_total_programs
            
            diff_indices = range(prog_idx, prog_idx+1)
            
            # first program, can't go back one program to compute diff dict
            if max_num_programs == 0 or is_final_program:
                # diff_dict = {
                #     'small_add': 0,
                #     'large_add': 0,
                #     'small_del': 0,
                #     'large_del': 0,
                #     'add_comment': 0,
                #     'add_func': 0,
                #     'color': 0,
                #     'number': 0,
                # } 
                # diff_dict = {
                #     'color_changes': 0,
                #     'number_changes': 0,
                #     'comment_additions': 0,
                #     'function_additions': 0,
                #     'lines_added': 0,
                #     'lines_deleted': 0,
                #     'lines_modified': 0,
                #     'edit_categories': {
                #         'large_add': 0,
                #         'small_add': 0,
                #         'large_del': 0,
                #         'small_del': 0,
                #         'modify_existing': 0,
                #     }
                # } 
                # sequential_edit_distance = 0
                
                # if before any code, first (next) program isn't backtracking; also can't backtracking in next program when no more code left
                is_backtrack = False
            else:
                # will get edit dict for indices max_num_programs-1, ie if max_num_programs=1, will get edit dict at index 0
                # diff_dict = trace.get_edit_dict(diff_indices=diff_indices, do_print=(do_print))
                
                # get whether or not about to backtrack
                # 0th element is 0, 1st element is distance between second program's distance to end and first program's distance to end, etc.
                # so want to get item at max_num_programs (ie 1st element for first program)
                diff_dists = trace.get_backtracking_diff_dists(diff_indices=diff_indices)
                
                assert len(diff_dists) == 1
                is_backtrack = diff_dists[0] > 0
                
            if is_final_program:
                # if final program, set everything to 0 (then later set edit category to no-edits)
                diff_dict = {
                    'color_changes': 0,
                    'number_changes': 0,
                    'comment_additions': 0,
                    'function_additions': 0,
                    'lines_added': 0,
                    'lines_deleted': 0,
                    'lines_modified': 0,
                    'edit_categories': {
                        'large_add': 0,
                        'small_add': 0,
                        'large_del': 0,
                        'small_del': 0,
                        'modify_existing': 0,
                    }
                } 
                sequential_edit_distance = 0
                
            else:
            
                diff_dict = trace.get_clean_edit_dict(diff_indices=diff_indices, do_print=(do_print))
                sequential_edit_distance = trace.get_sequential_edit_distance(diff_indices=diff_indices)
                assert len(sequential_edit_distance) == 1
                sequential_edit_distance = sequential_edit_distance[0]
                
            # TODO: setting to 'CODE' so that don't try to create tensor for empty string; fix eventually to be ''?
            if max_num_programs == 0:
                start_text = 'CODE'
            elif condition_code_format == 'full':
                start_text = trace.get_cleaned_trace_text(max_num_programs=max_num_programs)
            elif condition_code_format == 'last':
                # take last program and treat as first
                curr_program = trace.matched_programs[prog_idx]
                curr_date = trace.matched_dates[prog_idx]
                # TODO: always format as CODE 1?
                start_text = f"CODE 1 ({curr_date}):\n{curr_program}"
            else:
                raise ValueError(f'Invalid condition code format: {condition_code_format}')
            
            num_remaining = num_total_programs - max_num_programs
            if is_final_program:
                assert num_remaining == 0
            
            if do_print:
                print('-'*20)
                
            if do_mask_title:
                title_to_use = ''
            else:
                title_to_use = title
            
            # use user_to_condition_on to get embedding (username is ground truth user associated with trace; have distinction because need username for key to standardize data across experiments)
             
            embedding, num_input_tokens = embedding_extractor.get_embedding(
                code=start_text,
                program_name=title_to_use,
                username=user_to_condition_on,
                do_print=do_print,
                do_return_num_input_tokens=True
            )
            
            # TODO: what to do when num input tokens >= 1024? want to chunk and get embedding from chunk, but for now skip
            if num_input_tokens >= 1024:
                num_ex_with_break_per_trace += 1
                # TODO: remove break if want to compute mean num ex with break
                break
            
            assert not trace.is_degenerate(), f'Trace {trace.title} is degenerate'
            
            if trace_idx == 0:
                print(embedding[:, :10])
                print(start_text)
                print(f'Max num programs: {max_num_programs}')
                print(f'Num remaining: {num_remaining}')
           
            last_program = trace.get_last_program()
            # last_program_execution_result, _, _, _ = code_executor.execute_code([last_program], timeout_seconds_execute=5, timeout_seconds_navigate=5, max_concurrent=15)
            # last_program_execution_result = last_program_execution_result[0]
            
            # remove the first dimension
            embedding = embedding[0]
            trace_embeddings.append(embedding)
            trace_start_texts.append(start_text)
           
            # TODO: what should time_remaining be for max_num_programs == 0? (condition on first time?)
            if max_num_programs == 0:
                time_remaining_mins = 0
                time_remaining_seconds = 0
                edit_distance_to_last_program = 0
            else:
                try: 
                    time_remaining_mins = trace.get_time_diff(trace.matched_dates[max_num_programs-1], trace.matched_dates[-1], unit='minutes')
                    time_remaining_seconds = trace.get_time_diff(trace.matched_dates[max_num_programs-1], trace.matched_dates[-1], unit='seconds')
                except Exception as e:
                    print(f'Error: {e}')
                    num_issues += 1
                    # if not None, need to check why else this would be happening
                    assert trace.matched_dates[-1] == 'None', f'trace.matched_dates[-1] is not None: {trace.matched_dates[-1]}'
                    print(f'trace.matched_dates[-1] is None: {trace.matched_dates[-1]}')
                    print(f'trace.raw_trace_text: {trace.raw_trace_text}')
                    time_remaining_mins = 0
                    time_remaining_seconds = 0
                edit_distance_to_last_program = trace._get_edit_distance(trace.matched_programs[max_num_programs-1], trace.matched_programs[-1])
            
            if key in keys:
                breakpoint()
            keys.add(key)
            
            metadata = {
                'key': key,
                'program_idx': max_num_programs,
                'program_idx_normalized': max_num_programs / num_total_programs,
                'is_halfway_finished': max_num_programs >= num_total_programs / 2,
                'num_remaining': num_remaining,
                'num_total_programs': num_total_programs,
                'is_final_program': is_final_program,
                'start_text': start_text,
                'trace': trace,
                'diff_dict': diff_dict,
                'sequential_edit_distance': sequential_edit_distance,
                'is_backtrack': is_backtrack,
                'last_program': last_program,
                'username': username,
                'username_to_condition_on': user_to_condition_on,
                'program_name': title,
                'time_remaining_mins': time_remaining_mins,
                'time_remaining_seconds': time_remaining_seconds,
                'edit_distance_to_last_program': edit_distance_to_last_program,
                # 'last_program_is_correct': last_program_execution_result,
            }
            for key in ['color_changes', 'number_changes', 'comment_additions', 'function_additions', 'lines_added', 'lines_deleted', 'lines_modified']:
                metadata[key] = diff_dict[key]
                is_nonzero = metadata[key] != 0
                metadata[f'{key}_is_nonzero'] = is_nonzero
                
            edit_categories = diff_dict['edit_categories']
            # get nonzero edit category
            edit_categories_nonzero = [k for k, v in edit_categories.items() if v != 0]
            if is_final_program:
                # TODO: hackily setting to no-edits when can't compute edit category
                metadata['edit_category'] = 'no-edits'
            else:
                assert len(edit_categories_nonzero) == 1, f'Expected 1 nonzero edit category, got {len(edit_categories_nonzero)}'
                metadata['edit_category'] = edit_categories_nonzero[0]
                
               
            if do_print: 
                print_dict(metadata, keys_to_skip=['trace', 'diff_dict']) 
            trace_metadata.append(metadata)
            
        if num_ex_with_break_per_trace > 0:
            num_traces_with_break += 1
        else:
            embeddings.extend(trace_embeddings)
            metadata_list.extend(trace_metadata)
            start_texts.extend(trace_start_texts)
            
        # num_ex_with_break_list.append(num_ex_with_break_per_trace)
    
    # stack embeddings and targets so that they are two dimensional; embeddings is a list of 2D arrays
    embeddings = np.stack(embeddings)
    print(embeddings.shape)
    print(f'Code 0:\n{start_texts[0]}')
    print('-'*20)
    print(f'Code 1:\n{start_texts[1]}')
    print('-'*20)
    print(f'Code 2:\n{start_texts[2]}')
    print('-'*20)
    print(f'Embedding 0: {embeddings[0, :10]}')
    print(f'Embedding 1: {embeddings[1, :10]}')
    print(f'Embedding 2: {embeddings[2, :10]}')
    
    print('\n\n')
    print(f'Num traces with break: {num_traces_with_break}')
    # print(f'Mean num ex with break: {np.mean(num_ex_with_break_list)}')
    
    print(f'Num issues: {num_issues}')
    
    
    return embeddings, metadata_list

def get_data(train_traces, train_users, train_users_to_condition_on, test_traces, test_users, test_users_to_condition_on, embedding_extractor, condition_code_format='full', do_mask_title=False):
    
    print('Extracting embeddings and metadata for test traces...')
    X_test, test_metadata = extract_embeddings_and_metadata(test_traces, test_users, test_users_to_condition_on, embedding_extractor, condition_code_format=condition_code_format, do_mask_title=do_mask_title)
    # Extract embeddings and targets
    print('Extracting embeddings and metadata for train traces...')
    X_train, train_metadata = extract_embeddings_and_metadata(train_traces, train_users, train_users_to_condition_on, embedding_extractor, condition_code_format=condition_code_format, do_mask_title=do_mask_title)
   

    return X_train, train_metadata, X_test, test_metadata

def format_data(X_train, train_metadata, X_test, test_metadata, metric_name):
    # format data for probe experiment
    # X_train is a list of embeddings
    # train_metadata is a list of metadata
    # X_test is a list of embeddings
    # test_metadata is a list of metadata
    print('\n')
    print(f'Formatting data for metric: {metric_name}') 
  
    # if metric_name == 'is_halfway_finished':
    #     y_train = [metadata['program_idx_normalized'] >= 0.5 for metadata in train_metadata]
    #     y_test = [metadata['program_idx_normalized'] >= 0.5 for metadata in test_metadata]
    if metric_name == 'program_name':
        unique_program_names = [] 
        for metadata in train_metadata:
            unique_program_names.append(metadata['program_name'])
            
        for metadata in test_metadata:
            unique_program_names.append(metadata['program_name'])
            
        # counter
        # won't always be exactly the same bc inputs created for each program idx
        print(f'Counter of 20 most common program names across train and test: {Counter(unique_program_names).most_common(20)}')
        # breakpoint()
        program_name_to_idx = {program_name: idx for idx, program_name in enumerate(unique_program_names)}
        y_train = [program_name_to_idx[metadata['program_name']] for metadata in train_metadata]
        y_test = [program_name_to_idx[metadata['program_name']] for metadata in test_metadata]
    else:
        y_train = [metadata[metric_name] for metadata in train_metadata]
        y_test = [metadata[metric_name] for metadata in test_metadata]
        
    # get y values from metadata
    print(f'y_test: {y_test[:10]}')
    print(f'y_train: {y_train[:10]}')
    
    # first code
    # print(f'First metric: {y_test[0]}')
    # print(f'Matched dates: {test_metadata[0]["trace"].matched_dates}')
    # print(f'First code: {test_metadata[0]["trace"]}')
    
    # stack
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
   
    print('Done.') 
    return X_train, y_train, X_test, y_test

def run_regression_probe_experiment(X_train, y_train, X_test, y_test, title='None', do_show_plot=True, model_type='ridge', seed=0):
    """Train a model to predict remaining attempts and evaluate its performance."""
    
    set_seed(seed)
    
    # Train model
    if model_type == 'mlp':
        model = MLPRegressor(max_iter=5000, learning_rate_init=0.001, verbose=False, batch_size=64, early_stopping=True,hidden_layer_sizes=(100,100))
    # elif model_type == 'ridge':
    #     model = Ridge(alpha=10000)
    elif model_type == 'ridgecv':
        model = RidgeCV(cv=None, alphas=[1e3, 1e4, 1e5, 1e6, 1e7])
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'sgd':
        model = SGDRegressor(max_iter=5000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=100, eta0=0.0001)
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    probe = LinearProbe(model=model, method='regression')
    probe.train(X_train, y_train)
    
    # Evaluate
    print(f'Evaluating {model_type}...')
    train_results = probe.evaluate(X_train, y_train)
    test_results = probe.evaluate(X_test, y_test)
    
    for (name, result) in [('test', test_results), ('train', train_results)]:
        print('-'*30)
        print(name.upper())
        for key, value in result.items():
            if 'pred' in key or 'true' in key or key == 'squared_errors':
                print(f'{key}: {value[:10]}')
            else:
                print(f'{key}: {value}')
    test_mean_baseline_results = probe.evaluate_mean_baseline(y_train, y_test)
    train_mean_baseline_results = probe.evaluate_mean_baseline(y_train, y_train)
    
    return train_results, test_results, train_mean_baseline_results, test_mean_baseline_results

def run_classification_probe_experiment(X_train, y_train, X_test, y_test, title='None', do_show_plot=True, model_type='logistic',seed=0):
    """Train a model to predict remaining attempts and evaluate its performance."""
    
    set_seed(seed)
    
    # Train model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=5000, penalty='l2', solver='lbfgs')
    elif model_type == 'logisticcv':
        model = LogisticRegressionCV(cv=None, max_iter=1000)
    elif model_type == 'mlp':
        model = MLPClassifier(max_iter=5000, learning_rate_init=0.001, verbose=False, batch_size=64, early_stopping=True,hidden_layer_sizes=(100,100))
    elif model_type == 'ridge':
        model = RidgeClassifier()
    elif model_type == 'ridgecv':
        model = RidgeClassifierCV(cv=None, alphas=[1e3, 1e4, 1e5, 1e6, 1e7])
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    probe = LinearProbe(model=model, method='classification')
    probe.train(X_train, y_train)
    
    # Evaluate
    train_results = probe.evaluate(X_train, y_train)
    test_results = probe.evaluate(X_test, y_test)
    print(f'\nEvaluating majority baseline...')
    majority_baseline_results = probe.evaluate_majority_baseline(y_train, y_test)
    # plot confusion matrix (test_results['confusion_matrix'])
    
    return train_results, test_results, majority_baseline_results

def get_fig_axs(metrics, width_multiplier=1, height_multiplier=1.5, num_per_row=4):
    num_rows = len(metrics) // num_per_row
    if len(metrics) % num_per_row != 0:
        num_rows += 1
    num_cols = min(num_per_row, len(metrics))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(width_multiplier * num_cols * 3, height_multiplier * num_rows * 2))
    if num_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    return fig, axs

def get_mixed_splits(X_train, train_metadata, X_test, test_metadata, seed=0):
    print('Mixing train and test data...')
    # combine train and test
    X_data = np.concatenate([X_train, X_test], axis=0)
    train_metadata = train_metadata + test_metadata
    
    test_size = len(test_metadata) / (len(train_metadata) + len(test_metadata))
    
    print(f'Test size: {test_size}')
    # create train/test split
    X_train, X_test, train_metadata, test_metadata = train_test_split(X_data, train_metadata, test_size=test_size, random_state=seed)
    
    train_users = [metadata['username'] for metadata in train_metadata]
    test_users = [metadata['username'] for metadata in test_metadata]
    
    # get overlap
    overlap_users = set(train_users) & set(test_users)
    print(f'Overlapping users between train and test: {len(overlap_users)}')
    
    return X_train, train_metadata, X_test, test_metadata

def visualize_regression_results(all_X_train, all_train_metadata, all_X_test, all_test_metadata, fig_title=None, seeds=[0], program_idx='all', model_type='ridge', train_out_file=None, test_out_file=None, do_mix_users=True):
    
    # if len(seeds) > 1:
    #     raise NotImplementedError('Multiple seeds not supported for regression')
    metrics = [
        'num_remaining',
        'time_remaining_mins',
        'time_remaining_seconds',
        'edit_distance_to_last_program',
        'num_total_programs',
        # 'program_idx',
        'sequential_edit_distance',
        'program_idx_normalized',
        'color_changes', 'number_changes', 'comment_additions', 'function_additions', 'lines_added', 'lines_deleted', 'lines_modified'
        # 'max_num_programs', 
        # 'small_add', 'large_add', 'small_del', 'large_del', 'add_comment', 'add_func', 'color', 'number',
        ]
    train_fig, train_axs = get_fig_axs(metrics, width_multiplier=1.5, height_multiplier=2, num_per_row=4)
    train_axs = train_axs.flatten()
    test_fig, test_axs = get_fig_axs(metrics, width_multiplier=1.5, height_multiplier=2, num_per_row=4)
    test_axs = test_axs.flatten()
    
    # set to invisible
    for ax in train_axs:
        ax.set_visible(False)
    for ax in test_axs:
        ax.set_visible(False)
    pbar = tqdm(enumerate(metrics), total=len(metrics), desc='Running regression probe')
    results_by_metric = {}
    for metric_idx, metric_name in pbar:
        train_ax = train_axs[metric_idx]
        test_ax = test_axs[metric_idx]
        
        # if program_idx is not None, only plot for that program_idx
        if program_idx != 'all':
            new_X_train = []
            new_train_metadata = []
            new_X_test = []
            new_test_metadata = []
            num_skipped_train = 0
            num_skipped_test = 0
            for (X, metadata) in tqdm(zip(all_X_train, all_train_metadata), total=len(all_X_train), desc='Filtering train'):
                if program_idx == 'all_but_last':
                    if not metadata['is_final_program']:
                        new_X_train.append(X)
                        new_train_metadata.append(metadata)
                    else:
                        num_skipped_train += 1
                else:
                    assert isinstance(program_idx, int), 'program_idx must be an int if not all'
                    if metadata['program_idx'] == program_idx:
                        new_X_train.append(X)
                        new_train_metadata.append(metadata)
                    else:
                        num_skipped_train += 1
                    
        
            for (X, metadata) in tqdm(zip(all_X_test, all_test_metadata), total=len(all_X_test), desc='Filtering test'):
                if program_idx == 'all_but_last':
                    if not metadata['is_final_program']:
                        new_X_test.append(X)
                        new_test_metadata.append(metadata)
                    else:
                        num_skipped_test += 1
                else:
                    assert isinstance(program_idx, int), 'program_idx must be an int if not all'
                    if metadata['program_idx'] == program_idx:
                        new_X_test.append(X)
                        new_test_metadata.append(metadata)
                    num_skipped_test += 1
                  
            print(f'len(new_train_metadata): {len(new_train_metadata)}')  
            print(f'len(new_test_metadata): {len(new_test_metadata)}')
            X_train = np.array(new_X_train)
            train_metadata = new_train_metadata
            X_test = np.array(new_X_test)
            test_metadata = new_test_metadata
           
            # if num_skipped_train == 7787:
            #     breakpoint() 
            
            print(f'Num skipped train: {num_skipped_train}')
            print(f'Num skipped test: {num_skipped_test}')
       
        else:
            X_train = all_X_train
            train_metadata = all_train_metadata
            X_test = all_X_test
            test_metadata = all_test_metadata
        
        test_results_by_seed = []
        train_mean_baseline_results_by_seed = []
        test_mean_baseline_results_by_seed = []
        train_results_by_seed = []
        shuffled_test_results_by_seed = []
        for seed in seeds:
            pbar.set_description(f'Running regression probe {metric_name}, seed {seed}...')
            
            if do_mix_users:
                try:
                    temp_X_train, temp_train_metadata, temp_X_test, temp_test_metadata = get_mixed_splits(X_train, train_metadata, X_test, test_metadata, seed=seed)
                except Exception as e:
                    print(f'Error: {e}')
                    breakpoint()
            else:
                temp_X_train, temp_train_metadata, temp_X_test, temp_test_metadata = X_train, train_metadata, X_test, test_metadata
            temp_X_train, temp_y_train, temp_X_test, temp_y_test = format_data(temp_X_train, temp_train_metadata, temp_X_test, temp_test_metadata, metric_name=metric_name)
                
            print(f'temp_X_train: {temp_X_train.shape}')
            print(f'temp_y_train: {temp_y_train.shape}')
            print(f'temp_X_test: {temp_X_test.shape}')
            print(f'temp_y_test: {temp_y_test.shape}')
        
            train_results, test_results, train_mean_baseline_results, test_mean_baseline_results = run_regression_probe_experiment(temp_X_train, temp_y_train, temp_X_test, temp_y_test, title=None, do_show_plot=False, seed=seed, model_type=model_type)
            
            shuffled_X_train = temp_X_train[np.random.permutation(len(temp_X_train))]
            
            print(f'\nRunning shuffled...')
            _, shuffled_test_results, _, _ = run_regression_probe_experiment(shuffled_X_train, temp_y_train, temp_X_test, temp_y_test, title=None, do_show_plot=False, seed=seed, model_type=model_type)
            # print(test_results)
            test_results_by_seed.append(test_results)
            train_mean_baseline_results_by_seed.append(train_mean_baseline_results)
            test_mean_baseline_results_by_seed.append(test_mean_baseline_results)
            train_results_by_seed.append(train_results)
            shuffled_test_results_by_seed.append(shuffled_test_results)
            # Make predictions
            # y_test_pred = test_results['y_pred']
            # y_train_pred = train_results['y_pred']
            
            # y_test_true = test_results['y_true']
            # y_train_true = train_results['y_true']
        
            
            
        # print(test_results_by_seed)
        # mean_results_across_seeds = {key: np.mean([results[key] for results in test_results_by_seed]) for key in test_results_by_seed[0].keys()}
        # sem_results_across_seeds = {key: stats.sem([results[key] for results in test_results_by_seed]) for key in test_results_by_seed[0].keys()}
        # mean_baseline_results_across_seeds = {key: np.mean([results[key] for results in mean_baseline_results_by_seed]) for key in mean_baseline_results_by_seed[0].keys()}
        # sem_baseline_results_across_seeds = {key: stats.sem([results[key] for results in mean_baseline_results_by_seed]) for key in mean_baseline_results_by_seed[0].keys()}
        
        # since only one seed, just take first result
        # mean_results_across_seeds = test_results_by_seed[0]
        # sem_results_across_seeds = {}
        # mean_baseline_results_across_seeds = mean_baseline_results_by_seed[0]
        # sem_baseline_results_across_seeds = {}
       
        test_results_across_seeds = {}
        train_results_across_seeds = {}
        train_mean_baseline_results_across_seeds = {}
        test_mean_baseline_results_across_seeds = {}
        shuffled_test_results_across_seeds = {}
        
        for m in ['mse', 'r2', 'correlation']:
            temp_ms = [results[m] for results in test_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            test_results_across_seeds[f'{m}_mean'] = mean_m
            test_results_across_seeds[f'{m}_sem'] = sem_m
            
            temp_ms = [results[m] for results in shuffled_test_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            shuffled_test_results_across_seeds[f'{m}_mean'] = mean_m
            shuffled_test_results_across_seeds[f'{m}_sem'] = sem_m
            
            temp_ms = [results[m] for results in train_mean_baseline_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            train_mean_baseline_results_across_seeds[f'{m}_mean'] = mean_m
            train_mean_baseline_results_across_seeds[f'{m}_sem'] = sem_m
            
            temp_ms = [results[m] for results in test_mean_baseline_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            test_mean_baseline_results_across_seeds[f'{m}_mean'] = mean_m
            test_mean_baseline_results_across_seeds[f'{m}_sem'] = sem_m
            
            temp_ms = [results[m] for results in train_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            train_results_across_seeds[f'{m}_mean'] = mean_m
            train_results_across_seeds[f'{m}_sem'] = sem_m
            
        # assert len(test_results_by_seed) == 1
        # if len(seeds) == 1:
        seed = seeds[0]
        # plot for first seed
        test_result = test_results_by_seed[seed]
        # train_mean_baseline_result = train_mean_baseline_results_by_seed[seed]
        # test_mean_baseline_result = test_mean_baseline_results_by_seed[seed]
        # shuffled_test_result = shuffled_test_results_by_seed[seed]
        train_result = train_results_by_seed[seed]
        
        for (name, results, axs) in [('test', test_result, test_axs), ('train', train_result, train_axs)]:
            # Plot predictions vs actual
            
            ax = axs[metric_idx]
            ax.scatter(results['y_true'], results['y_pred'], alpha=0.2)
            # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            title = f'{metric_name} ({name})\nMSE: {results["mse"]:.3f}\nR²: {results["r2"]:.3f}\nCorrelation: {results["correlation"]:.3f}'
            ax.set_title(title)
            
            axs[metric_idx].set_title(title)
            ax.set_visible(True)
        
        title = metric_name + f'\nMSE: {test_result["mse"]:.3f}\nR²: {test_result["r2"]:.3f}\nseed: {seed}'
    
        if fig_title is not None:
            # set figtitle for each fig
            for fig, title, out_file in zip([train_fig, test_fig], [f'train {fig_title}', f'test {fig_title}'], [train_out_file, test_out_file]):
                fig.suptitle(title)
                fig.tight_layout()
                if out_file is not None:
                    if not os.path.exists(os.path.dirname(out_file)):
                        os.makedirs(os.path.dirname(out_file))
                    print(f'Saving to {out_file}')
                    fig.savefig(out_file)
        plt.show()
        
        
        results_by_metric[metric_name] = {
                    'test': test_results_across_seeds,
                    'train': train_results_across_seeds,
                    # 'sem':sem_results_across_seeds,
                    'train_mean_baseline':train_mean_baseline_results_across_seeds,
                    'test_mean_baseline':test_mean_baseline_results_across_seeds,
                    'shuffled_test':shuffled_test_results_across_seeds,
                    # 'sem_baseline':sem_baseline_results_across_seeds
                } 
                
    
    return results_by_metric
    
def visualize_classification_results(all_X_train, all_train_metadata, all_X_test, all_test_metadata, fig_title=None, seeds=[0], program_idx='all', model_type='logistic', out_file=None, do_mix_users=True, metrics=None):
    if metrics is None:
        metrics = [
            'program_name',
            'is_final_program', 'is_halfway_finished',
           'small_add_is_nonzero', 'large_add_is_nonzero', 'small_del_is_nonzero', 'large_del_is_nonzero', 'add_comment_is_nonzero', 'add_func_is_nonzero', 'color_is_nonzero', 'number_is_nonzero',
           ]
    
    # if len(seeds) > 1:
    #     raise NotImplementedError('Multiple seeds not supported for classification')

    width_multiplier = 2
    height_multiplier = 3.5
    num_per_row = 4
    num_rows = len(metrics) // num_per_row
    if len(metrics) % num_per_row != 0:
        num_rows += 1
    num_cols = num_per_row  
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(width_multiplier * num_cols * 3, height_multiplier * num_rows * 2))
    axs = axs.flatten()
    # set to invisible
    for ax in axs:
        ax.set_visible(False)
        
    results_by_metric = {}
    pbar = tqdm(enumerate(metrics), total=len(metrics), desc='Running classification probe')
    for metric_idx, metric_name in pbar:
        ax = axs[metric_idx]
        
        
        # if program_idx is not None, only plot for that program_idx
        if program_idx != 'all':
            new_X_train = []
            new_train_metadata = []
            new_X_test = []
            new_test_metadata = []
            for (X, metadata) in tqdm(zip(all_X_train, all_train_metadata), total=len(all_X_train), desc='Filtering train'):
                # make sure not last program
                if program_idx == 'all_but_last':
                    if not metadata['is_final_program']:
                        new_X_train.append(X)
                        new_train_metadata.append(metadata)
                else:
                    assert isinstance(program_idx, int), 'program_idx must be an int if not all'
                    if metadata['program_idx'] == program_idx:
                        new_X_train.append(X)
                        new_train_metadata.append(metadata)
                    
        
            for (X, metadata) in tqdm(zip(all_X_test, all_test_metadata), total=len(all_X_test), desc='Filtering test'):
                # make sure not last program
                if program_idx == 'all_but_last':
                    if not metadata['is_final_program']:
                        new_X_test.append(X)
                        new_test_metadata.append(metadata)
                else:
                    assert isinstance(program_idx, int), 'program_idx must be an int if not all'
                    if metadata['program_idx'] == program_idx:
                        new_X_test.append(X)
                        new_test_metadata.append(metadata)
                    
            X_train = np.array(new_X_train)
            train_metadata = new_train_metadata
            X_test = np.array(new_X_test)
            test_metadata = new_test_metadata
            
        # consider all program indices
        else:
            X_train = all_X_train
            train_metadata = all_train_metadata
            X_test = all_X_test
            test_metadata = all_test_metadata
                    
        
        test_results_by_seed = []
        majority_baseline_results_by_seed = []
        shuffled_test_results_by_seed = []
        
        for seed in seeds:
            set_seed(seed)
            if do_mix_users:
                temp_X_train, temp_train_metadata, temp_X_test, temp_test_metadata = get_mixed_splits(X_train, train_metadata, X_test, test_metadata, seed=seed)
                
            else:
                temp_X_train, temp_train_metadata, temp_X_test, temp_test_metadata = X_train, train_metadata, X_test, test_metadata
            temp_X_train, temp_y_train, temp_X_test, temp_y_test = format_data(temp_X_train, temp_train_metadata, temp_X_test, temp_test_metadata, metric_name=metric_name)
                
            print(f'temp_X_train: {temp_X_train.shape}')
            print(f'temp_y_train: {temp_y_train.shape}')
            print(f'temp_X_test: {temp_X_test.shape}')
            print(f'temp_y_test: {temp_y_test.shape}')
            
            
            pbar.set_description(f'Running classification probe for metric {metric_name}, seed {seed}, model {model_type}...')
            _, test_results, majority_baseline_results = run_classification_probe_experiment(temp_X_train, temp_y_train, temp_X_test, temp_y_test, title=None, do_show_plot=False, seed=seed, model_type=model_type)
            
           
            # set seed
            set_seed(seed)
            # shuffle X_train
            shuffled_X_train = temp_X_train[np.random.permutation(len(temp_X_train))]
            _, test_results_shuffled, _ = run_classification_probe_experiment(shuffled_X_train, temp_y_train, temp_X_test, temp_y_test, title=None, do_show_plot=False, seed=seed, model_type=model_type)
            
            for (name, results) in [('test', test_results), ('majority_baseline', majority_baseline_results), ('shuffled_test', test_results_shuffled)]:
                print(name.upper())
                for key, value in results.items():
                    if key not in ['y_true', 'y_pred', 'confusion_matrix']:
                        print(f'{name} {key}: {value}')
            
            
            shuffled_test_results_by_seed.append(test_results_shuffled)
            
            # print(test_results)
            test_results_by_seed.append(test_results)
            majority_baseline_results_by_seed.append(majority_baseline_results)
        # print(test_results_by_seed)
        
        # mean_results_across_seeds = {key: np.mean([results[key] for results in test_results_by_seed]) for key in test_results_by_seed[0].keys()}
        # sem_results_across_seeds = {key: stats.sem([results[key] for results in test_results_by_seed]) for key in test_results_by_seed[0].keys()}
        # mean_majority_baseline_results_across_seeds = {key: np.mean([results[key] for results in majority_baseline_results_by_seed]) for key in majority_baseline_results_by_seed[0].keys()}
        # sem_majority_baseline_results_across_seeds = {key: stats.sem([results[key] for results in majority_baseline_results_by_seed]) for key in majority_baseline_results_by_seed[0].keys()}
        
        
        
        # since only one seed, just take first result
        # assert len(test_results_by_seed) == 1
        # test_result = test_results_by_seed[0]
        # majority_baseline_result = majority_baseline_results_by_seed[0]
        # shuffled_test_result = shuffled_test_results_by_seed[0]
       
        test_results_across_seeds = {}
        shuffled_test_results_across_seeds = {}
        majority_baseline_results_across_seeds = {}
       
        for m in ['accuracy', 'f1', 'precision', 'recall']:
            temp_ms = [results[m] for results in test_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            test_results_across_seeds[f'{m}_mean'] = mean_m
            test_results_across_seeds[f'{m}_sem'] = sem_m
            
            temp_ms = [results[m] for results in shuffled_test_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            shuffled_test_results_across_seeds[f'{m}_mean'] = mean_m
            shuffled_test_results_across_seeds[f'{m}_sem'] = sem_m
            
            temp_ms = [results[m] for results in majority_baseline_results_by_seed]
            mean_m = np.mean(temp_ms)
            sem_m = stats.sem(temp_ms)
            majority_baseline_results_across_seeds[f'{m}_mean'] = mean_m
            majority_baseline_results_across_seeds[f'{m}_sem'] = sem_m
            
        
       
        # shuffled_accuracy = shuffled_test_result['accuracy']
        # shuffled_f1 = shuffled_test_result['f1']
        # shuffled_precision = shuffled_test_result['precision']
        # shuffled_recall = shuffled_test_result['recall']
     
        ax = axs[metric_idx]
        
        # visualize for first seed
        if len(seeds) == 1:
            test_result = test_results_by_seed[0]
            accuracy = test_result['accuracy_mean']
            f1 = test_result['f1_mean']
            precision = test_result['precision_mean']
            recall = test_result['recall_mean']
            
            if test_result['num_classes'] <= 10:
                ax.imshow(test_result['confusion_matrix'], cmap='Blues')
                # annotate values
                for i in range(test_result['confusion_matrix'].shape[0]):
                    for j in range(test_result['confusion_matrix'].shape[1]):
                        ax.text(j, i, test_results['confusion_matrix'][i, j], ha='center', va='center', color='black')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                title = metric_name + f'\nAccuracy: {accuracy:.3f}\nF1: {f1:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
                axs[metric_idx].set_title(title)
                ax.set_visible(True)
            else:
                print(f'Skipping confusion matrix for {metric_name} because it has {test_result["num_classes"]} classes')
                
        results_by_metric[metric_name] = {
            'test': test_results_across_seeds,
            'majority_baseline':majority_baseline_results_across_seeds,
            'shuffled_test':shuffled_test_results_across_seeds,
        }
    if fig_title is not None:
        plt.suptitle(fig_title)
    # only plot if len(seeds) is 1, otherwise plotting gets messsed up
    if len(seeds) == 1:
        plt.tight_layout()
        plt.show()
        if out_file is not None:
            print(f'Saving to {out_file}')
            plt.savefig(out_file)
    return results_by_metric
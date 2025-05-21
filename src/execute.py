import sys
import asyncio
import os
import pickle
import time
import threading
from playwright.async_api import async_playwright, TimeoutError
from src.visualize import *
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import psutil

# DEBUG = False 
DEBUG = False

# custom exception
class CustomCodeException(Exception):
    pass

class CustomTimeoutError(Exception):
    pass

def is_exception_timeout_ok(code_snippet):
    # check if 'await' is in the code snippet (with space/word boundaries)
    # use regex
    if re.search(r'\bawait\b', code_snippet):
        return True 
    # forever loop but no stop()
    if re.search(r'\bforever\b', code_snippet) and not re.search(r'\stop()\b', code_snippet):
        return True
    else:
        return False
def is_exception_error_ok(result):
    # check if errors has 'ERR_FILE_NOT_FOUND' bc this error should be ok for execution? TODO: confirm
    # check if all errors have 'ERR_FILE_NOT_FOUND'
    if all(['ERR_FILE_NOT_FOUND' in error for error in result['errors']]):
        return True
    else:
        return False

def check_result_validity(result):
    had_errors = result['errors'] != []
    # no errors, didn't reach end, and no 'await' in the code snippet
    failed_to_reach_end = not had_errors and not result['end_reached'] and not is_exception_timeout_ok(result['code_snippet'])
    
    # had errors, reached end
    reached_end_but_had_errors = had_errors and result['end_reached'] and not is_exception_error_ok(result)
    if failed_to_reach_end:
        print('-'*20)
        print(f"Code snippet: {result['code_snippet']}")
        print(f"Failed to reach end but had no errors.")
        print(f"End reached: {result['end_reached']}")
        print(f"Errors: {result['errors']}")
        print(f"Logs: {result['logs']}")
        print(f"File: {result['file']}")
        print(f'Timed out: {result["timeout"]}')
        print(f'Is timeout ok: {is_exception_timeout_ok(result["code_snippet"])}')
        print(f'Try: {result["try_idx"]}. Is max try: {result["is_max_try"]}')
        # breakpoint()
        # Don't raise exception, is likely instance of code actually timing out
        if result['is_max_try']:
            print('Max try reached')
            print(f'Try idx: {result["try_idx"]}')
            print(f'Not raising exception, returning as is with no errors')
            assert len(result['errors']) == 0, f"Errors: {result['errors']}"
            assert not result['timeout'], f"Timeout: {result['timeout']}"
        else:
            raise CustomCodeException(f"Failed to reach end but had no errors. Logs: {result['logs']}")
        # if result['timeout']:
        #     raise CustomCodeException(f"Failed to reach end but had no errors. Logs: {result['logs']}")
        # else:
        #     print("Didn't time out so not raising exception")
        print('-'*20)
    # TODO: not raising exception for this bc sometimes still reaches end
    if reached_end_but_had_errors:
        print('-'*20)
        print(f"Reached end but had errors.")
        print(f"End reached: {result['end_reached']}")
        print(f"Errors: {result['errors']}")
        print(f"Logs: {result['logs']}")
        print(f"File: {result['file']}")
        # print("Not raising exception")
        print('-'*20)
        # raise CustomCodeException(f"Reached end but had errors: {result['errors']}")
    if result['timeout']:
        print('-'*20)
        print(f"Timeout: {result['timeout']}")
        print(f'Is timeout ok: {is_exception_timeout_ok(result["code_snippet"])}')
        print(f"Code snippet: {result['code_snippet']}")
        if is_exception_timeout_ok(result["code_snippet"]):
            print("Not raising exception bc timeout is ok")
        else:
            raise CustomTimeoutError(f"Timeout: {result['timeout']}")
        print('-'*20)
    return True


### HELPER UTILS IN CASE RUNNING CODE FROM JUPYTER NOTEBOOK ###
class RunThread(threading.Thread):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))

def run_async(func, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func(*args, **kwargs))

#################################################################

class CodeExecutor:
    def __init__(self, cache_file='cache/code_execution.pkl'):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
    
    def load_cache(self, max_num_tries=3):
        print(f"Loading cache from {self.cache_file}")
        cache = {}
        num_tries = 0
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, 'rb') as f:
                        cache = pickle.load(f)
                    break
                except Exception as e:
                    print(f"Error loading cache: {e}")
                    num_tries += 1
                    
                    time.sleep(1)
                    print(f"Retrying... ({num_tries}/{max_num_tries})")
                
                    if num_tries >= max_num_tries:
                        print(f"Failed to load cache after {num_tries} tries. Starting with empty cache.")
                        break
        else:
            # check if directory exists
            if not os.path.exists(os.path.dirname(self.cache_file)):
                os.makedirs(os.path.dirname(self.cache_file))
                print(
                    f'Cache directory "{os.path.dirname(self.cache_file)}" created.'
                )

        self.cache.update(cache)
        print(f"Loaded {len(cache)} entries from cache.")
        
    def write_cache(self, max_num_tries=3, do_print=True):
        num_tries = 0
        print(f"Writing cache to {self.cache_file}")
        while True:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                break
            except Exception as e:
                print(f"Error writing cache: {e}")
                num_tries += 1
                
                time.sleep(1)
                print(f"Retrying... ({num_tries}/{max_num_tries})")
                
                if num_tries >= max_num_tries:
                    print(f"Failed to write cache after {num_tries} tries.")
                    break
        if do_print:
            print(f"Wrote {len(self.cache)} entries in cache to {self.cache_file}")
   
    def get_cache_key(self, code_snippet, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent, language):
        return (code_snippet, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent, language)
    
    
    def execute_code(self, code_snippets, timeout_seconds_execute=5, timeout_seconds_navigate=5, max_concurrent=15, do_use_cache=True, language='coffeescript'):
        
        MAX_VALID_CONCURRENT = 100 
        assert max_concurrent <= MAX_VALID_CONCURRENT, f"max_concurrent is greater than {MAX_VALID_CONCURRENT}; to avoid errors in code execution caused by too many concurrent processes, please set max_concurrent to {MAX_VALID_CONCURRENT}"
        
        # for each code snippet, check if it's in the cache; store ones that need to be run
        to_run_code_snippets = []
        to_run_indices = []
        results = [None] * len(code_snippets)
        for i, prog in enumerate(code_snippets):
            # TODO: add max_concurrent to cache key?
            cache_key = self.get_cache_key(prog, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent, language)
            if cache_key in self.cache and do_use_cache:
                results[i] = self.cache[cache_key]
            else:
                to_run_code_snippets.append(prog)
                to_run_indices.append(i)
                
        # print(f"Checking {len(to_run_code_snippets)} code snippets...")
        
        # create temp html files to run
        html_files = []
        for i, prog in enumerate(to_run_code_snippets):
            # convert current datetime to string (including microseconds)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            html_file = f'temp_{i}_{current_time}.html'
            
            # make sure html_file doesn't already exist
            temp_idx = 0
            while os.path.exists(html_file):
                print(f'File {html_file} already exists. Adding number to end.')
                # add a number to the end of the file name
                html_file = html_file.split('.')[0] + f'_{temp_idx}.html'
                temp_idx += 1
            
                
            prog_with_log = f'{prog}\nconsole.log("END_REACHED");'
            
            write_html_file(prog_with_log, html_file, language=language)
            html_files.append(html_file)
       
        # TODO: look at https://stackoverflow.com/questions/55409641/asyncio-run-cannot-be-called-from-a-running-event-loop-when-using-jupyter-no to check on jupyter 
        # results, execution_time = asyncio.run(process_files(html_files, timeout, max_concurrent))
        
        assert len(html_files) == len(to_run_code_snippets), f"len(html_files) {len(html_files)} != len(to_run_code_snippets) {len(to_run_code_snippets)}"
      
        # first run coffeescript
        files_and_metadata = [
            {
                'file': html_files[i],
                'order': i,
                'code_snippet': to_run_code_snippets[i] 
            }
            for i in range(len(html_files))
        ]
        
        new_results, _ = run_async(process_files, files_and_metadata, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent)
        # breakpoint()
        # delete temp html files
        for file_idx, file in enumerate(html_files):
            assert file == new_results[file_idx]['file'], f"File {file} does not match new_results[file_idx]['file'] {new_results[file_idx]['file']}"
            os.remove(file)
            
        
        # update cache with new results
        for i, result in zip(to_run_indices, new_results):
            code_snippet = code_snippets[i]
            cache_key = self.get_cache_key(code_snippet, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent, language)
            self.cache[cache_key] = result
            
        # self.write_cache()
        
        # return all results, including cached ones
        for (orig_idx, result) in zip(to_run_indices, new_results):
            results[orig_idx] = result
            
        # make sure no 'None' results are returned
        if None in results:
            print("Some results are missing. Please check the code.")
        return results
    
    def parse_result(self, result, is_rerun=False):
        passed_with_no_errors = result['errors'] == []
        end_reached = result['end_reached']
        # with errors and reached end
        reached_end_but_had_errors = not passed_with_no_errors and end_reached and not is_exception_error_ok(result)
        # with no errors and didn't reach end
        failed_to_reach_end = passed_with_no_errors and not end_reached and not is_exception_timeout_ok(result['code_snippet'])
        new_result = result.copy()
        reached_end_with_no_errors = passed_with_no_errors and end_reached
        new_result.update({
            'passed_with_no_errors': passed_with_no_errors,
            'end_reached': end_reached,
            'reached_end_but_had_errors': reached_end_but_had_errors,
            'failed_to_reach_end': failed_to_reach_end,
            'is_rerun': is_rerun,
            'reached_end_with_no_errors': reached_end_with_no_errors
        })
        return new_result
    
    def get_execute_results(self, code_snippets, timeout_seconds_execute=5, timeout_seconds_navigate=5, max_concurrent=15, do_print=False, language='coffeescript', retry_language='javascript'):
        """Gets whether passed with no errors for each code snippet"""

        try:
            # only use cache on first attempt
            results = self.execute_code(code_snippets, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent, do_use_cache=True, language=language)
            
            parsed_results = [self.parse_result(result) for result in results]
            
            if do_print:
                print("Results:")
                for i, r in enumerate(parsed_results):
                    print(f"{i+1}. passed with no errors={r['passed_with_no_errors']}")
                    print(f"Errors: {r['errors']}")
            
            
            
            # if any didn't have end_reached, re-run them
            results_to_rerun = [i for i, r in enumerate(parsed_results) if not r['reached_end_with_no_errors']]
            if len(results_to_rerun) > 0:
                print(f"Some results failed to reach end with no errors. Re-running {len(results_to_rerun)} results with {retry_language}...")
                results_to_rerun_code_snippets = [code_snippets[i] for i in results_to_rerun]
                results_to_rerun_results = self.execute_code(results_to_rerun_code_snippets, timeout_seconds_execute, timeout_seconds_navigate, max_concurrent, do_use_cache=False, language=retry_language)
                
                # take the max of the results
                for new_idx, orig_idx in enumerate(results_to_rerun):
                    # update parsed_results with current result (don't need to explicitly take max bc only reran for results that didn't reach end with no errors)
                    parsed_results[orig_idx] = self.parse_result(results_to_rerun_results[new_idx], is_rerun=True)
                        
            passed_with_no_errors_results = [r['passed_with_no_errors'] for r in parsed_results]
            end_reached_results = [r['end_reached'] for r in parsed_results]

            failed_to_reach_end_results = [r['failed_to_reach_end'] for r in parsed_results]
            reached_end_but_had_errors_results = [r['reached_end_but_had_errors'] for r in parsed_results]
                

            
            # make sure no True values in failed_to_reach_end_results
            if True in failed_to_reach_end_results:
                num_true = sum(failed_to_reach_end_results)
                print(f"{num_true} results failed to reach end but had no errors.")
                print(f"Passed with no errors results: {passed_with_no_errors_results}")
                print(f"End reached results: {end_reached_results}")
                print(f"Failed to reach end results: {failed_to_reach_end_results}")
                print("Code snippets that failed to reach end:")
                for i, result in enumerate(results):
                    if failed_to_reach_end_results[i]:
                        print(f"{code_snippets[i]}")
                        print('-'*20)
                breakpoint()
                # raise CustomCodeException(f"{num_true} results failed to reach end but had no errors.")
                # breakpoint()
            return passed_with_no_errors_results, end_reached_results, failed_to_reach_end_results, reached_end_but_had_errors_results
        
        except Exception as e:
            breakpoint()

async def check_console_errors(file_and_metadata, timeout_seconds_execute=5, timeout_seconds_navigate=5, try_idx=0, is_max_try=False):
    total_timeout = timeout_seconds_execute + timeout_seconds_navigate
    
    # print(f'Checking console errors...')
    # Convert to absolute path if it's not already
    html_path = file_and_metadata['file']
    if not html_path.startswith('file://') and not html_path.startswith('http'):
        # Get absolute path
        abs_path = os.path.abspath(html_path)
        # Convert to file:// URL format
        url = f"file://{abs_path}"
    else:
        url = html_path
    
    results = {
        "file": html_path,
        "code_snippet": file_and_metadata['code_snippet'],
        "order": file_and_metadata['order'],
        "errors": [],
        "warnings": [],
        "logs": [],
        "successfully_executed": False,
        "timeout": False,
        "network_idle_reached": False,
        "end_reached": False,
        "try_idx": try_idx,
        "is_max_try": is_max_try
    }
    
    async with async_playwright() as playwright:
        # Launch Chrome browser using your installed binary
        browser = await playwright.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True
        )
        
        # Create a new page
        page = await browser.new_page()
        
        # Set default timeout for all operations
        page.set_default_timeout(total_timeout * 1000)  # Convert to milliseconds
        
        # Set up console message handling
        async def handle_console(msg):
            msg_type = msg.type
            text = msg.text
            
            if msg_type == "error":
                results["errors"].append(text)
                results["logs"].append(f"ERROR: {text}")
                # terminate early by raising timeout error
                # raise CustomTimeoutError(f"Error: Manually raised timeout error because found error in console.")
            elif msg_type == "warning":
                results["warnings"].append(text)
                results["logs"].append(f"WARNING: {text}")
            elif msg_type == "log":
                if DEBUG: print(f"{msg_type.upper()}: {text}")
                results["logs"].append(f"{msg_type.upper()}: {text}")
                if text == "END_REACHED":
                    results["end_reached"] = True
            else:
                results["logs"].append(f"{msg_type.upper()}: {text}")
        
        # Listen for console events
        page.on("console", handle_console)
        
        # Listen for page errors (uncaught exceptions)
        async def handle_page_error(error):
            error_message = f"UNCAUGHT EXCEPTION: {error}"
            if DEBUG: print(f'UNCAUGHT EXCEPTION: {error}')
            results["errors"].append(error_message)
            results["logs"].append(error_message)
            
        page.on("pageerror", handle_page_error)
        
        # handle networkidle
        # TODO: not sure if this works
        async def handle_network_idle():
            results["logs"].append(f"Network idle reached.")
            print(f'Network idle reached')
            results["network_idle_reached"] = True
        
        page.on("networkidle", handle_network_idle)
        
        try:
            # Navigate to the file URL and wait for network idle with timeout
            if DEBUG: print(f'Navigating to {url}')
            time_before_goto = time.time()
            await page.goto(url, wait_until="networkidle", timeout=timeout_seconds_navigate * 1000)
            results["logs"].append("Page loaded completely")
            time_after_goto = time.time()
            if DEBUG: print(f'Page loaded completely')
            if DEBUG: print(f'Time taken to load page: {time_after_goto - time_before_goto} seconds')
            # check if networkidle is reached
            # if page.is_network_idle():
            #     results["logs"].append(f"Network idle reached.")
            #     results["network_idle_reached"] = True
            
            # Wait a bit for any delayed scripts to run
            # num_delay_seconds = timeout_seconds
            if DEBUG: print(f'Waiting for {timeout_seconds_execute} seconds')
            # await asyncio.wait_for(asyncio.sleep(timeout_seconds_execute), timeout= total_timeout)
            await asyncio.wait_for(asyncio.sleep(timeout_seconds_execute), timeout=total_timeout)
            if DEBUG: print(f'Done waiting')
            results["successfully_executed"] = True
               
        # except CustomTimeoutError as e:
        #     print(e)
        #     pass 
        except (asyncio.TimeoutError, TimeoutError) as e:
            results["timeout"] = True
            results["logs"].append(f"TIMEOUT: The page took longer than {timeout_seconds_execute} seconds to load + {timeout_seconds_navigate} seconds to navigate")
        finally:
            await browser.close()
        
        # if file_and_metadata["order"] == 9:
        #     breakpoint()
         
        if DEBUG: print(f'Order: {file_and_metadata["order"]}; Code snippet: {file_and_metadata["code_snippet"][:20]}; html_path: {file_and_metadata["file"]}; Errors: {results["errors"]}; Logs: {results["logs"]}')
           
        # print error
        # if results["errors"]:
        #     print(f"Errors: {results['errors']}")
        # if results["warnings"]:
        #     print(f"Warnings: {results['warnings']}")
        # if results["logs"]:
        #     print(f"Logs: {results['logs']}")
            
        return results

async def process_files(files_and_metadata, timeout_seconds_execute=5, timeout_seconds_navigate=5, max_concurrent=15):
    """Process multiple files concurrently with a limit on max concurrent jobs and show a progress bar"""
    start_time = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    running_tasks = set()

    async def process_with_semaphore(file_and_metadata):
        async with semaphore:
            # Print number of running tasks
            running_tasks.add(file_and_metadata['file'])
            if DEBUG: print(f"START: {file_and_metadata['file']} | Running tasks: {len(running_tasks)}")
            max_tries = 2 
            num_tries = 0
            temp_timeout_seconds_execute = timeout_seconds_execute
            temp_timeout_seconds_navigate = timeout_seconds_navigate
            temp_timeout_total = timeout_seconds_navigate + timeout_seconds_execute + 30
            while num_tries < max_tries:
                try:
                    # Wrap the task in a timeout and catch all exceptions
                    is_max_try = num_tries >= max_tries - 1
                    result = await asyncio.wait_for(
                        check_console_errors(file_and_metadata, temp_timeout_seconds_execute, temp_timeout_seconds_navigate, try_idx=num_tries, is_max_try=is_max_try),
                        timeout=temp_timeout_total  # generous upper bound
                    )
                    if not check_result_validity(result):
                        raise CustomCodeException(f"Result is invalid.")
                    break
                except Exception as e:
                    print(f"EXCEPTION in {file_and_metadata['file']}.")
                    print(f"Error: {e.__class__.__name__}: {e}")
                    temp_timeout_seconds_execute += 100
                    temp_timeout_seconds_navigate += 100
                    temp_timeout_total = temp_timeout_seconds_navigate + temp_timeout_seconds_execute + 30
                    num_tries += 1
                    if num_tries > max_tries:
                        raise CustomCodeException(f"Reached max tries ({max_tries}). Number of tries: {num_tries}. Try making max_concurrent smaller or increasing timeout_seconds_execute and timeout_seconds_navigate")
                    else:
                        print(f'Retrying with timeout_seconds_execute: {temp_timeout_seconds_execute}; timeout_seconds_navigate: {temp_timeout_seconds_navigate}')
                    # print(f'Code snippet: {file_and_metadata["code_snippet"]}')
                # result = None
            # except Exception as e:
            #     result = {"file": file_and_metadata['file'], "error": str(e)}
            #     breakpoint()
            #     break
            # if result is None:
            #     print(f"FAILED: {file_and_metadata['file']} timed out after {max_task_retries} attempts.")
            #     # result = {"file": file_and_metadata['file'], "error": f"Task timed out after {max_task_retries} attempts"}
            #     print(f'Result is None; should not happen')
            #     breakpoint()
            if DEBUG: print(f"END: {file_and_metadata['file']} | Running tasks: {len(running_tasks)-1}")
            running_tasks.remove(file_and_metadata['file'])
            return result

    tasks = [process_with_semaphore(f) for f in files_and_metadata]

    # Use asyncio.as_completed to update the progress bar as each task finishes
    results = []
    did_break = False
    e = None
    if len(tasks) > 0:
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for coro in asyncio.as_completed(tasks):
                # need to catch exceptions here, otherwise the progress bar will not update
                try:
                    result = await coro
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f'Breaking')
                    # TODO: this part is still not working
                    did_break = True
                    breakpoint()
                    # break out of asyncio.as_completed, just 'break' doesn't work
                    # TODO: hackily setting result to None to break out of asyncio.as_completed bc breaking doesn't seem to work; setting to None will lead to an error outside of this function
                    result = None
    if did_break:
        print(f'Raising error')
        raise CustomCodeException(f"Error")

    results = sorted(results, key=lambda x: x["order"])
    end_time = time.time()
    execution_time = end_time - start_time
    return results, execution_time

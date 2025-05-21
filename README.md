# Overview
This is the public release for the paper "Learning to Model Student Learning with 3.5 Million Program Traces". 

For the current submission, we release a small sample of our dataset and code that evaluats different metrics of program traces. 
Concretely, we release 50 traces for the "snowman" assignment from the ground truth data, synthetic model, last state model, and trace mode. 

Example Usage:

### Calculate BLEU with respect to ground truth user rollouts

```
>>> from src.metrics import *                                                                                                                                                                                                                                                                                                                                                              
>>> all_tc = TraceCollection()                                                                                                                                                                                      
>>> all_tc.load_trace_collection_json("rollouts/all_snowman")                                                                                                                                                       
>>> last_tc = TraceCollection()                                                                            
>>> last_tc.load_trace_collection_json("rollouts/last_snowman")                                            
>>> user_tc = TraceCollection()                                                                            
>>> user_tc.load_trace_collection_json("rollouts/user_snowman")                                            
>>> all_tc.reference_bleu(user_tc)
0.7180487441182128
>>> last_tc.reference_bleu(user_tc)                                                                        
0.7266377850039696
```

### Printing the last state of a program

```
>>> print(all_tc.traces[-1].matched_programs[-1])
speed 30
dot thistle, 5000
dot snow, 100
fd 80
dot snow, 75
fd 60
dot snow, 50
bk 40
dot black, 10
bk 25
dot black, 10
bk 25
dot black, 10
fd 70
lt 90
fd 10
dot blue, 10
rt 180
fd 20
dot blue, 10
rt 120
fd 10
pen black, 5
lt 120, 10
lt 60
fd 10
pu()
lt 120
```

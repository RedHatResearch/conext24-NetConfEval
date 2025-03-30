#!/bin/bash

while getopts r:m:f: flag
do
  case "${flag}" in
    r) n_runs=${OPTARG};;
    m) model=${OPTARG};;
    f) fn_call_support=${OPTARG};;
  esac
done

if [ -z "$n_runs" ]; then
  echo 'You need to specify the number of runs.' >&2
  exit 1
fi

if [ -z "$model" ]; then
  echo 'You need to specify the model identifier.' >&2
  exit 1
fi

if [ -z "$fn_call_support" ]; then
  echo 'You need to specify model function calling support.' >&2
  exit 1
fi

if [ "$fn_call_support" != "0" ] && [ "$fn_call_support" != "1" ]
then
    echo "Invalid value for 'fn_call_support': $fn_call_support." >&2
    exit 1
fi

# Translation
python3 netconfeval/step_1_formal_spec_translation.py --n_run $n_runs --model $model --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
python3 netconfeval/step_1_formal_spec_translation.py --n_run $n_runs --model $model --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
python3 netconfeval/step_1_formal_spec_translation.py --n_run $n_runs --model $model --policy_types reachability --batch_size 1 2 5 10 20 50 100

# Conflict Detection
python3 netconfeval/step_1_formal_spec_conflict_detection.py --model $model --policy_types reachability waypoint loadbalancing --n_run $n_runs --batch_size 1 3 11 33

# Function Call
if [ "$fn_call_support" == "1" ]
then
  python3 netconfeval/step_1_function_call.py --n_run $n_runs --model $model --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33
  python3 netconfeval/step_1_function_call.py --n_run $n_runs --model $model --policy_types reachability waypoint --batch_size 1 2 5 10 25 50
  python3 netconfeval/step_1_function_call.py --n_run $n_runs --model $model --policy_types reachability --batch_size 1 2 5 10 20 50 100
else
  python3 netconfeval/step_1_function_call.py --n_run $n_runs --model $model --policy_types reachability waypoint loadbalancing --batch_size 1 3 11 33 --adhoc
  python3 netconfeval/step_1_function_call.py --n_run $n_runs --model $model --policy_types reachability waypoint --batch_size 1 2 5 10 25 50 --adhoc
  python3 netconfeval/step_1_function_call.py --n_run $n_runs --model $model --policy_types reachability --batch_size 1 2 5 10 20 50 100 --adhoc
fi

# Code Generation
python3 netconfeval/step_2_code_gen.py --model $model --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10
python3 netconfeval/step_2_code_gen.py --model $model --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10 --feedback
python3 netconfeval/step_2_code_gen.py --model $model --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10 --prompts no_detail
python3 netconfeval/step_2_code_gen.py --model $model --n_run $n_runs --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10 --prompts no_detail --feedback

# Low-Level Configurations
python3 netconfeval/step_3_low_level.py --n_run $n_runs --model $model --mode none
python3 netconfeval/step_3_low_level.py --n_run $n_runs --model $model --mode full
python3 netconfeval/step_3_low_level.py --n_run $n_runs --model $model --mode idx
python3 netconfeval/step_3_low_level.py --n_run $n_runs --model $model --mode rag --rag_chunk_size 9000

NetConfEval: Can LLMs Facilitate Network Configuration?

## What is it?

We present a set of benchmarks (NetConfEval) to examine the effectiveness of different models in facilitating and automating network configuration described in our paper "[NetConfEval: Can LLMs Facilitate Network Configuration?](https://doi.org/10.1145/3656296)".

[📜 Paper](https://doi.org/10.1145/3656296) - [🤗 Hugging Face Dataset](https://huggingface.co/datasets/NetConfEval/NetConfEval)

## Installation
Make sure to use Python 3.10 or later. Install this repository and install all the packages.
``` bash
git clone git@github.com:RedHatResearch/NetConfEval.git
virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt
```

To run experiments with OpenAI models, export your [OPENAI_API_KEY](https://platform.openai.com/api-keys) in the environment:
```bash
export OPENAI_API_KEY="YOUR OPENAI KEY"
```

To run experiments with Huggingface models, install the additional packages and login with your Huggingface account with your [Access Token](https://huggingface.co/settings/tokens).
```bash
pip install -r requirements-hf.txt
huggingface-cli login
```

To run the **Generating Low-level Configurations** experiment, you need to install [Docker](https://docs.docker.com/engine/install/) on your system.

## Usage

### Quick Start

To run the benchmark on a specific model, simply run:
```bash
./run_benchmark.sh -r <n_runs> -m <model_id> -f <0/1>
```

The `<n_runs>` specifies the number of iterations for each experiment. We suggest a value of `5`.

The `<model_id>` is the model identifier used in the `netconfeval/common/model_configs.py` file.
Check [Support for New Models](#support-for-new-models) for instructions on how to include a new model.

The `-f` flag specifies if the model supports native parallel function calling (e.g., GPT-4-Turbo) and it should be used in the benchmark (value to `1`).
Otherwise, the benchmark relies on ad-hoc function calling (value to `0`).

### Experiments Details

#### Translating High-Level Requirements to a Formal Specification Format
This test evaluates LLMs' ability to translate network operators' requirements into a formal specification. For instance, the input information can be converted into a simple data structure to specify the reachability, waypoints, and load-balancing policies in a network.

Here is an example of the experiment. We use `gpt-4-1106` to translate multiple requirements into a formal specification made of the three policies, with a batch size of 3:
```bash
python3 step_1_formal_spec_translation.py --n_run 1 --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 3
```

The experiment results will be stored in the directory named `results_spec_translation` by default.

#### Translating High-Level Requirements to Functions/API Calls
This test evaluates the ability of LLMs' to translate natural language requirements into corresponding function/API calls, which is a common task in network configuration since many networks employ SDN, where a software controller can manage the underlying network via direct API calls.

To translate a few requirements into multiple function calls (```add_reachability(), add_waypoint(), add_load_balance()```) in parallel, run:

```bash
python3 step_1_function_call.py --n_runs 1 --model gpt-4-1106 --policy_types reachability waypoint loadbalancing --batch_size 3
```

**Ad-hoc function calling.** Since most models do not support *parallel function calling* natively, we customize the input prompt to evaluate ad-hoc function calling.

To run the experiment:

```bash
python3 step_1_function_call.py --n_runs 1 --model gpt-4 --policy_types reachability waypoint loadbalancing --batch_size 3 --adhoc
```

The experiment results will be stored in the directory named `results_function_call` by default.

#### Developing Routing Algorithms
Traffic engineering is a critical yet complex problem in network management, particularly in large networks. Our experiment asks the models to create functions that compute routing paths based on specific network requirements (the shortest path, reachability, waypoint, load balancing). 

To run the experiment:

```bash
python3 step_2_code_gen.py --model gpt-4-1106 --n_runs 1 --policy_types shortest_path reachability waypoint loadbalancing --n_retries 10
```

The experiment results will be stored in the directory named `results_code_gen` by default.

#### Generating Low-level Configurations
This experiment explores the problem of transforming high-level requirements into detailed, low-level configurations suitable for installation on network devices. We handpicked four network scenarios publicly available in the [Kathará Network Emulator repository](https://github.com/KatharaFramework/Kathara-Labs). The selection encompasses the most widespread protocols and consists of two OSPF networks (one single-area network and one multi-area network), a RIP network, a BGP network featuring a basic peering between two routers, and a small fat-tree datacenter network running a made-up version of RIFT. All these scenarios (aside from RIFT) leverage FRRouting as the routing suite. 

After installing [Docker](https://docs.docker.com/engine/install/), you can run:

```bash
python3 step_3_low_level.py --n_runs 1 --model gpt-4-turbo --mode rag --rag_chunk_size 9000
```

The experiment results will be stored in the directory named `results_low_level` by default.

## Support for New Models
We rely on LangChain to provide a common interface to access different model APIs.
You can add new supported models in the `netconfeval/common/model_configs.py` file.
We currently support:
* OpenAI models (`'type': 'openai'`)
* HuggingFace models (`'type': 'HF'`) through a custom LangChain-compatible class (see `netconfeval/foundation/langchain/hf.py`)
* Ollama models (`'type': 'Ollama'`) (thanks to @RobertoLorusso) 

To add a model, just add a new Dict element to the `model_configurations` Dict, by providing a unique key for it.
The new model key is then automatically visible using the `--model` command line parameter of the `.py` tests of the benchmarks.

### OpenAI Models
The OpenAI model Dict element contains the following keys:
```python
{
    'type': 'openai', # The type of the model, in this case 'openai'
    'model_name': 'gpt-3.5-turbo-1106', # The model name taken by from OpenAI APIs
    'args': {   # A Dict containing parameters that are directly passed from LangChain to the underlying OpenAI object, can be empty
        'response_format': {'type': 'json_object'},
        'seed': 5000,
    }
}
```

Let's assume we want to add `gpt-4-32k-0613` model, we just append the following entry:
```python
model_configurations = {
    ...
    'gpt-4-32k-0613': {
        'type': 'openai',
        'model_name': 'gpt-4-32k-0613',
        'args': {} # Pass additional parameters if needed
    }
}
```

### Ollama Models
The Ollama model Dict contains the following keys:
```python
{
    'type': 'Ollama', # The type of the model, in this case 'Ollama'
    'model_name': 'llama3:8b-instruct-fp16', # The model name taken from Ollama library
    'num_predict': 4096 # Max output length
}
```

### HuggingFace Models
The HuggingFace model Dict contains the following keys:
```python
{
    'type': 'HF', # The type of the model, in this case 'HF'
    'model_name': 'meta-llama/Llama-2-7b-chat-hf', # The model name taken from HuggingFace
    'prompt_builder': _build_llama2_prompt, # If the model requires a special prompt builder, you can pass the function reference here
    'max_length': 4096, # Max output length
    'use_quantization': False # Whether to use or no quantization
}
```

Let's assume we want to add a new HuggingFace model, for example LLama3, we just append the following entry:
```python
model_configurations = {
    ...
    'llama3-8b-instruct': {
        'type': 'HF',
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'prompt_builder': _build_llama3_prompt, # Implement the function
        'max_length': 4096,
        'use_quantization': False
    }
}
```

### Adding new model types
Aside from adding OpenAI, Ollama, and HuggingFace models, it is also possible to add new model types (for example Gemini by Google).

We will continuously improve support for different APIs, but if you want to contribute:
- Define a new `type`, coherent with the model types (e.g., `google` for Google models);
- Modify the `get_model_instance` function in `netconfeval/common/model_configs.py` and add specific model loading.

## Citing our paper
If you use NetConfEval, please cite our paper:

```bibtex
@article{netconfeval,
    author = {Wang, Chanjie and Scazzariello, Mariano and Farshin, Alireza and Ferlin, Simone and Kosti\'{c}, Dejan and Chiesa, Marco},
    title = {NetConfEval: Can LLMs Facilitate Network Configuration?},
    year = {2024},
    issue_date = {June 2024},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {2},
    number = {CoNEXT2},
    url = {https://doi.org/10.1145/3656296},
    doi = {10.1145/3656296},
    journal = {Proc. ACM Netw.},
    month = {june},
    articleno = {7},
    numpages = {25},
}
```

## Help
If you have any questions regarding our code or the paper, you can contact [Changjie Wang](https://www.kth.se/profile/changjie) (changjie at kth.se) and/or [Mariano Scazzariello](https://www.ri.se/en/person/mariano-scazzariello) (mariano.scazzariello at ri.se).

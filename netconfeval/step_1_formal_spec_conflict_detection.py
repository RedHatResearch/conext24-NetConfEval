import argparse
import json
import os
import sys
import time
from json import JSONDecodeError

from deepdiff import DeepDiff
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.formatters.formatters import step_1_input_formatter, step_1_conflict_formatter
from netconfeval.foundation.step.chain_step import ChainStep
from netconfeval.common.model_configs import model_configurations, get_model_instance
from netconfeval.common.utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(model_configurations.keys()),
                        required=True)
    parser.add_argument('--n_runs', type=int, required=False, default=5)
    parser.add_argument("--policy_file", type=str, required=False,
                        default=os.path.join("..", "assets", "step_1_policies.csv"))
    parser.add_argument('--batch_size', type=int, nargs="+", required=False,
                        default=[1, 2, 5, 10, 20, 25, 50, 100])
    parser.add_argument('--policy_types', choices=["reachability", "waypoint", "loadbalancing"],
                        required=False, nargs='+', default=["reachability"])
    parser.add_argument(
        '--results_path', type=str, default=os.path.join("..", "results_conflict_detection")
    )
    parser.add_argument('--combined', action='store_true', required=False)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    policy_types = SortedSet(args.policy_types)
    if "reachability" not in policy_types:
        logging.error("`reachability` is not in policy_types! Aborting...")
        exit(1)
    elif "loadbalancing" in policy_types and "waypoint" not in policy_types:
        logging.error("You cannot require for `loadbalancing` without `waypoint`! Aborting...")
        exit(1)

    if args.combined:
        from netconfeval.prompts.step_1_reachability_waypoint_load import SETUP_PROMPT, FUNCTION_PROMPT, \
            ASK_FOR_RESULT_PROMPT
    else:
        from netconfeval.prompts.step_1_conflict_detection import SETUP_PROMPT, FUNCTION_PROMPT, ASK_FOR_RESULT_PROMPT

    os.makedirs(args.results_path, exist_ok=True)

    results_time = time.strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(
        os.path.abspath(
            os.path.join(
                args.results_path,
                f"log-{args.model}{'-combined' if args.combined else ''}-{'_'.join(policy_types)}-conflict-{results_time}.log"
            )
        )
    )
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(logging.WARNING)
    logging.root.addHandler(file_handler)

    dataset = load_csv(args.policy_file, policy_types)

    llm_step_1 = get_model_instance(args.model)

    n_policy_types = len(policy_types)
    max_n_requirements = max(args.batch_size) * n_policy_types
    w = None

    filename = f"result-{args.model}{'-combined' if args.combined else ''}-{'_'.join(policy_types)}-conflict-{results_time}.csv"

    with open(os.path.join(args.results_path, filename), 'w') as f:
        for it in range(0, args.n_runs):
            logging.info(f"Performing iteration n. {it + 1}...")
            samples = pick_sample(max_n_requirements, dataset, it, policy_types)

            for batch_size in args.batch_size:
                flag_conflict = True
                logging.info(f"Performing experiment "
                             f"with {batch_size * n_policy_types} batch size (iteration n. {it + 1})...")
                chunk_samples = list(chunk_list(samples, batch_size * n_policy_types))
                if len(chunk_samples) == 1:
                    chunk_new = copy.deepcopy(chunk_samples[0])
                    chunk_samples.append(chunk_new)

                for i, sample in enumerate(chunk_samples):
                    logging.info(f"Performing experiment with {batch_size * n_policy_types} "
                                 f"batch size on chunk {i} (iteration n. {it + 1})...")

                    result_row = {
                        'model_error': '',
                        'format_error': '',
                        'batch_size': batch_size,
                        'n_policy_types': n_policy_types,
                        'max_n_requirements': max_n_requirements,
                        'iteration': it,
                        'chunk': i,
                        'time': 0,
                        'total': 0,
                        'success': 0,
                        'fail': 0,
                        'wrong': 0,
                        'accuracy': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_cost': 0,
                        'diff': '',
                        'conflict_exist': flag_conflict,
                        'conflict_detect': False,
                    }

                    if w is None:
                        w = csv.DictWriter(f, result_row.keys())
                        w.writeheader()

                    if flag_conflict:
                        insert_conflict(sample)
                    flag_conflict = not flag_conflict

                    expected_spec = transform_sample_to_expected(sample)
                    human_language = convert_to_human_language(sample)

                    logging.warning(f"==== RUN #{it + 1} (CHUNK #{i + 1}) - BATCH: {batch_size}*{n_policy_types} ====")
                    logging.warning("Expected Result: " + json.dumps(expected_spec, indent=4))
                    logging.warning("Human Translation: " + " ".join(human_language))

                    skip_compare = False
                    start_time = time.time()
                    if model_configurations[args.model]['type'] in ['HF', 'Ollama']:
                        # Combine all system prompts with a new line separator
                        combined_system_prompt = f"{SETUP_PROMPT}\n{FUNCTION_PROMPT}\n{ASK_FOR_RESULT_PROMPT}"
                        messages = [
                            ("system", combined_system_prompt),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("user", "{input}"),
                        ]
                    elif model_configurations[args.model]['type'] == 'openai':
                        messages = [
                            ("system", SETUP_PROMPT),
                            ("system", FUNCTION_PROMPT),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            ("system", ASK_FOR_RESULT_PROMPT),
                        ]
                    else:
                        raise Exception(
                            f"Type `{model_configurations[args.model]['type']}` for Model `{args.model}` not supported!"
                        )

                    prompt = ChatPromptTemplate.from_messages(messages)
                    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    chain_step = LLMChain(
                        llm=llm_step_1,
                        prompt=prompt,
                        verbose=False,
                        memory=memory
                    )
                    step_1 = ChainStep(
                        llm_chain=chain_step,
                        input_formatter=step_1_input_formatter,
                        output_formatter=step_1_conflict_formatter,
                    )

                    result = {}
                    output = ''
                    try:
                        if model_configurations[args.model]['type'] == 'openai':
                            with get_openai_callback() as cb:
                                status, output = step_1.process(' '.join(human_language))
                                result_row['prompt_tokens'] = cb.prompt_tokens
                                result_row['completion_tokens'] = cb.completion_tokens
                                result_row['total_cost'] = cb.total_cost
                        else:
                            status, output = step_1.process(' '.join(human_language))
                        logging.warning("Output: ", output)
                        if not status:
                            result_row['conflict_detect'] = True
                            result_row['diff'] = output
                            skip_compare = True
                        elif "args" in output:
                            result = output["args"][0]
                            print("result: ", result)
                        else:
                            result_row['format_error'] = str(output)
                            skip_compare = True

                    except JSONDecodeError as e:
                        result_row['format_error'] = str(e)
                        result = output
                        skip_compare = True
                    except Exception as e:
                        result_row['model_error'] = str(e)
                        skip_compare = True

                    logging.warning("LLM Result: " + str(result))
                    logging.warning("==================================================================")

                    if not skip_compare:
                        result_row['time'] = time.time() - start_time

                        new_result = copy.copy(result)
                        if "waypoint" in result:
                            new_result["waypoint"] = {}
                            for k, v in result["waypoint"].items():
                                new_result["waypoint"][k.replace(" ", "")] = v
                        if "loadbalancing" in result:
                            new_result["loadbalancing"] = {}
                            for k, v in result["loadbalancing"].items():
                                new_result["loadbalancing"][k.replace(" ", "")] = v

                        compare_result(expected_spec, new_result, result_row)
                        diff = DeepDiff(expected_spec, new_result, ignore_order=True)
                        result_row['diff'] = str(diff)

                    w.writerow(result_row)
                    f.flush()


if __name__ == "__main__":
    main(parse_args())

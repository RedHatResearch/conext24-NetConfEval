import argparse
import os
import sys
import time

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import get_openai_callback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from netconfeval.common.model_configs import model_configurations, get_model_instance
from netconfeval.common.utils import *
from netconfeval.formatters.formatters import step_2_input_formatter, step_2_output_formatter
from netconfeval.foundation.langchain.memory.conversation_latest_memory import ConversationLatestMemory
from netconfeval.foundation.step.chain_step import ChainStep
from netconfeval.verifiers.step_2_verifier_detailed import Step2VerifierDetailed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(model_configurations.keys()), required=True)
    parser.add_argument('--n_runs', type=int, required=False, default=5)
    parser.add_argument(
        '--policy_types',
        choices=["shortest_path", "reachability", "waypoint", "loadbalancing"],
        required=False, nargs='+', default=["shortest_path"]
    )
    parser.add_argument(
        '--prompts', choices=["basic", "no_detail"],
        required=False, default="basic"
    )
    parser.add_argument(
        '--results_path', type=str, default=os.path.join("..", "results_code_gen")
    )
    parser.add_argument('--feedback', action="store_true")
    parser.add_argument('--n_retries', type=int, default=10)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    if args.prompts == "basic":
        from netconfeval.prompts.step_2_basic import SETUP_PROMPT, INPUT_OUTPUT_PROMPT, INSTRUCTION_PROMPT, \
            FEEDBACK_CODE_GENERATION, ADD_REQUIREMENT_REACHABILITY_PROMPT, ADD_REQUIREMENT_WAYPOINT_PROMPT, \
            ADD_REQUIREMENT_LOADBALANCING_PROMPT, ASK_FOR_CODE_PROMPT
    if args.prompts == "no_detail":
        from netconfeval.prompts.step_2_no_detail import SETUP_PROMPT, INPUT_OUTPUT_PROMPT, INSTRUCTION_PROMPT, \
            FEEDBACK_CODE_GENERATION, ADD_REQUIREMENT_REACHABILITY_PROMPT, ADD_REQUIREMENT_WAYPOINT_PROMPT, \
            ADD_REQUIREMENT_LOADBALANCING_PROMPT, ASK_FOR_CODE_PROMPT

    with_feedback = args.feedback if args.feedback else False

    os.makedirs(args.results_path, exist_ok=True)

    results_time = time.strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(
        os.path.abspath(
            os.path.join(
                args.results_path,
                f"log-{args.model}-{'_'.join(args.policy_types)}-{args.prompts}-{'without_feedback' if not with_feedback else 'with_feedback'}-{results_time}.log"
            )
        )
    )

    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(logging.WARNING)
    logging.root.addHandler(file_handler)

    llm_step_2 = get_model_instance(args.model)

    w = None

    code_base_assets_path = os.path.abspath(os.path.join('..', 'assets', 'step_2_code_base'))
    tests_assets_path = os.path.abspath(os.path.join('..', 'assets', 'step_2_tests'))
    filename = f"result-{args.model}-{'_'.join(args.policy_types)}-{args.prompts}-{'without_feedback' if not with_feedback else 'with_feedback'}-{results_time}.csv"
    with open(os.path.join(args.results_path, filename), 'w') as f:
        for it in range(0, args.n_runs):
            for policy in args.policy_types:
                test_cases_path = "path_tests"

                code_path = os.path.join(code_base_assets_path, "computing_path_empty.py")
                add_requirement_prompts = ""

                if policy == "reachability":
                    code_path = os.path.join(code_base_assets_path, "computing_path_shortest_path.py")
                    add_requirement_prompts = ADD_REQUIREMENT_REACHABILITY_PROMPT
                    test_cases_path = "reachability_tests"
                elif policy == "waypoint":
                    code_path = os.path.join(code_base_assets_path, "computing_path_reachability.py")
                    add_requirement_prompts = ADD_REQUIREMENT_WAYPOINT_PROMPT
                    test_cases_path = "waypoint_tests"
                elif policy == "loadbalancing":
                    code_path = os.path.join(code_base_assets_path, "computing_path_shortest_path.py")
                    add_requirement_prompts = ADD_REQUIREMENT_LOADBALANCING_PROMPT
                    test_cases_path = "loadbalancing_tests"

                with open(code_path, "r") as f_code:
                    code = f_code.read()
                prompt = code + "\n" + add_requirement_prompts

                logging.warning(f"(Iteration n. {it}) performing code generation with {policy}...")

                result_row = {
                    'iteration': it,
                    'policy': policy,
                    'time': 0,
                    'feedback_num': 0,
                    'format_error_num': 0,
                    'syntax_error_num': 0,
                    'test_error_num': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_cost': 0,
                }

                if w is None:
                    w = csv.DictWriter(f, result_row.keys())
                    w.writeheader()

                if model_configurations[args.model]['type'] in ['HF', 'Ollama']:
                    combined_system_prompt = f"{SETUP_PROMPT}\n{ASK_FOR_CODE_PROMPT}"
                    combined_human_prompt = f"{INPUT_OUTPUT_PROMPT}\n{INSTRUCTION_PROMPT}\n{{input}}"
                    if with_feedback:
                        messages = [
                            ("system", combined_system_prompt),
                            ("user", combined_human_prompt),
                            MessagesPlaceholder(variable_name="chat_history"),
                        ]
                    else:
                        messages = [
                            ("system", combined_system_prompt),
                            ("user", combined_human_prompt),
                        ]
                elif model_configurations[args.model]['type'] == 'openai':
                    if with_feedback:
                        messages = [
                            ("system", SETUP_PROMPT),
                            ("human", INPUT_OUTPUT_PROMPT),
                            ("human", INSTRUCTION_PROMPT),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            ("system", ASK_FOR_CODE_PROMPT)
                        ]
                    else:
                        messages = [
                            ("system", SETUP_PROMPT),
                            ("human", INPUT_OUTPUT_PROMPT),
                            ("human", INSTRUCTION_PROMPT),
                            ("human", "{input}"),
                            ("system", ASK_FOR_CODE_PROMPT)
                        ]
                else:
                    raise Exception(
                        f"Type `{model_configurations[args.model]['type']}` for Model `{args.model}` not supported!"
                    )

                prompt_step_2 = ChatPromptTemplate.from_messages(messages)
                memory_step_2 = ConversationLatestMemory(memory_key="chat_history", return_messages=True)
                chain_step_2 = LLMChain(
                    llm=llm_step_2,
                    prompt=prompt_step_2,
                    verbose=True,
                    memory=memory_step_2,
                )

                step_2_verifier = Step2VerifierDetailed(
                    default_test_path=os.path.join(tests_assets_path, "path_tests")
                )

                step_2 = ChainStep(
                    llm_chain=chain_step_2,
                    verifier=step_2_verifier,
                    feedback_prompt=FEEDBACK_CODE_GENERATION if with_feedback else None,
                    feedback_retries=args.n_retries,
                    input_formatter=step_2_input_formatter,
                    output_formatter=step_2_output_formatter
                )

                start = time.time()

                if model_configurations[args.model]['type'] == 'openai':
                    with get_openai_callback() as cb:
                        try:
                            step_2.process(
                                {
                                    "extend": True,
                                    "input": prompt,
                                    "metadata": {"test_path": os.path.join(tests_assets_path, test_cases_path)}
                                }
                            )
                        except Exception as e:
                            logging.warning(e)

                        result_row['prompt_tokens'] = cb.prompt_tokens
                        result_row['completion_tokens'] = cb.completion_tokens
                        result_row['total_cost'] = cb.total_cost
                else:
                    try:
                        step_2.process(
                            {
                                "extend": True,
                                "input": prompt,
                                "metadata": {"test_path": os.path.join(tests_assets_path, test_cases_path)}
                            }
                        )
                    except Exception as e:
                        logging.warning(e)

                result_row['test_error_num'] = step_2.test_error
                result_row['syntax_error_num'] = step_2.syntax_error
                result_row['format_error_num'] = step_2.format_error
                result_row['feedback_num'] = step_2.failure_number

                end = time.time()
                result_row['time'] = end - start

                w.writerow(result_row)
                f.flush()


if __name__ == '__main__':
    main(parse_args())

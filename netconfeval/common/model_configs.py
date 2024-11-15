from typing import Any


def _build_llama2_prompt(messages):
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() == 'user':
            conversation.append(content)
        elif role.lower() == 'ai':
            conversation.append(f"{content}")
        elif role.lower() == 'function':
            raise ValueError('Llama 2 does not support function calls.')
        elif role.lower() == 'system':
            conversation.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
        else:
            raise ValueError(f"Invalid message role: {role}")

    return start_prompt + ''.join(conversation) + end_prompt


def _build_phind_prompt(messages):
    system_token = "### System Prompt \n"
    user_token = "\n ### User Message \n"
    assistant_token = "\n ### Assistant \n"

    system_conversation = []
    user_conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() == 'user':
            user_conversation.append(content)
        elif role.lower() == 'function' or role.lower() == 'ai':
            raise ValueError('Llama 2 does not support function and ai calls.')
        elif role.lower() == 'system':
            system_conversation.append(content)
        else:
            raise ValueError(f"Invalid message role: {role}")

    return system_token + ''.join(system_conversation) + user_token + ''.join(user_conversation) + assistant_token


def _build_mistral_instruct_prompt(messages):
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() in ['user', 'system']:  # 'user' and 'system' are treated the same.
            conversation.append(content)
        elif role.lower() in ['ai', 'function']:  # 'ai' and 'function' roles are not supported.
            raise ValueError(f"Mistral model does not support the role: {role}")
        else:
            raise ValueError(f"Unexpected role: {role}")

    # Assemble the prompt with the start and end tokens.
    prompt = start_prompt + ' '.join(conversation) + end_prompt

    return prompt


def _build_mistral_lite_prompt(messages):
    start_prompt = "<|prompter|> "
    end_prompt = " </s><|assistant|>"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() in ['user', 'system']:  # 'user' and 'system' are treated the same.
            conversation.append(content)
        elif role.lower() in ['ai', 'function']:  # 'ai' and 'function' roles are not supported.
            raise ValueError(f"Mistral model does not support the role: {role}")
        else:
            raise ValueError(f"Unexpected role: {role}")

    # Assemble the prompt with the start and end tokens.
    prompt = start_prompt + ' '.join(conversation) + end_prompt

    return prompt

def _build_qwen2_prompt(messages):
    
    start_turn = "<|im_start|>"
    end_turn = "<|im_end|>\n"
 

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() in ['user', 'system', 'assistant']:
            conversation.append(start_turn + role.lower() + '\n' + content + end_turn) 
        else:
            raise ValueError(f"Unexpected role: {role}")

   # Assemble the prompt with the start and end tokens and start the turn of assistant to prime the generation process
    prompt = ' '.join(conversation) + start_turn + "assistant\n" 

    return prompt 


def _build_llama3_prompt(messages):
    start_prompt = "<|begin_of_text|>"
    start_role = "<|start_header_id|>"
    end_role = "<|end_header_id|>"
    end_turn = "<|eot_id|>"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() in ['user', 'system', 'assistant']:
            conversation.append(start_role + role.lower() + end_role + content + end_turn ) # Example: <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        else:
            raise ValueError(f"Unexpected role: {role}")

   # Assemble the prompt with the start and end tokens and start the turn of assistant to prime the generation process
    prompt = start_prompt + ' '.join(conversation)  + '\n' + start_role + "assistant" + end_role
    print(prompt)
    return prompt


def get_model_instance(model_name: str) -> Any:

    if model_configurations[model_name]['type'] == 'HF':
        from netconfeval.foundation.langchain.chat_models.hf import ChatHF

        return ChatHF(
            model_name=model_configurations[model_name]['model_name'],
            max_length=model_configurations[model_name]['max_length'],
            use_quantization=model_configurations[model_name]['use_quantization'],
            prompt_func=model_configurations[model_name]['prompt_builder'],
        )
    
    elif model_configurations[model_name]['type'] == 'Ollama':
        from langchain_community.llms import Ollama

        return Ollama(model = model_configurations[model_name]['model_name'],
                      num_predict = model_configurations[model_name]['num_predict'],
                      num_gpu=-1
                     )
        
    elif model_configurations[model_name]['type'] == 'openai':
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model_name=model_configurations[model_name]['model_name'],
            model_kwargs=model_configurations[model_name]['args'],
        )
    else:
        raise Exception(f"Type `{model_configurations[model_name]['type']}` for model `{model_name}` not supported!")


model_configurations = {
    'gpt-3.5-turbo': {
        'model_name': 'gpt-3.5-turbo',
        'type': 'openai',
        'args': {}
    },
    'gpt-3.5-1106': {
        'model_name': 'gpt-3.5-turbo-1106',
        'type': 'openai',
        'args': {
            'response_format': {'type': 'json_object'},
            'seed': 5000,
        }
    },
    'gpt-3.5-0613': {
        'model_name': 'gpt-3.5-turbo-0613',
        'type': 'openai',
        'args': {}
    },
    'gpt-4': {
        'model_name': 'gpt-4',
        'type': 'openai',
        'args': {}
    },
    'gpt-4-turbo': {
        'model_name': 'gpt-4-turbo',
        'type': 'openai',
        'args': {}
    },
    'gpt-4-1106': {
        'model_name': 'gpt-4-1106-preview',
        'type': 'openai',
        'args': {
            'response_format': {'type': 'json_object'},
            'seed': 5000,
        }
    },

    #### Start Ollama models ####

    'llama3.1-ollama': {
        'type': 'Ollama',
        'model_name': 'llama3.1:8b-instruct-fp16',
        'num_predict': 4096
    },

    'llama3-ollama': {
        'type': 'Ollama',
        'model_name': 'llama3:8b-instruct-fp16',
        'num_predict': 4096
    },

    'neural-chat-ollama': {
        'type': 'Ollama',
        'model_name': 'neural-chat:7b-v3.3-fp16',
        'num_predict': 4096
    },

    # 4-bit quantization version 

    'llama3.1-4bit-ollama': {
        'type': 'Ollama',
        'model_name': 'llama3.1:latest',
        'num_predict': 4096
    },

    'llama3-4bit-ollama': {
        'type': 'Ollama',
        'model_name': 'llama3:latest',
        'num_predict': 4096
    },
      
    'neural-chat-4bit-ollama': {
        'type': 'Ollama',
        'model_name': 'neural-chat:latest',
        'num_predict': 4096
    },

    #### End Ollama models #####
    

    'qwen2.5-7b-instruct': {
        'type': 'HF',
        'model_name':'Qwen/Qwen2.5-7B-Instruct',
        'prompt_builder': _build_qwen2_prompt,
        'max_length': 4096,
        'use_quantization': False
    },

    'llama3-8b-instruct': {
        'type': 'HF',
        'model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'prompt_builder': _build_llama3_prompt,
        'max_length': 4096,
        'use_quantization': False
    },

    'llama3.1-8b-instruct': {
        'type': 'HF',
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'prompt_builder': _build_llama3_prompt,
        'max_length': 4096,
        'use_quantization': False
    },

    'llama2-7b-chat': {
        'model_name': 'meta-llama/Llama-2-7b-chat-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'llama2-13b-chat': {
        'model_name': 'meta-llama/Llama-2-13b-chat-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-7b-instruct': {
        'model_name': 'codellama/CodeLlama-7b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-13b-instruct': {
        'model_name': 'codellama/CodeLlama-13b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-34b-instruct': {
        'model_name': 'codellama/CodeLlama-34b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'use_quantization': True,
        'max_length': 4096,
        'type': 'HF'
    },
    'phind-34b-v2': {
        'model_name': 'Phind/Phind-CodeLlama-34B-v2',
        'prompt_builder': _build_phind_prompt,
        'use_quantization': True,
        'max_length': 4096,
        'type': 'HF'
    },
    'mistral-7b-instruct': {
        'model_name': 'mistralai/Mistral-7B-Instruct-v0.1',
        'prompt_builder': _build_mistral_instruct_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'mistral-lite': {
        'model_name': 'amazon/MistralLite',
        'prompt_builder': _build_mistral_lite_prompt,
        'max_length': 32768,
        'type': 'HF',
        'use_quantization': False
    },
}

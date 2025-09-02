import json
import logging
import re
import sys
import time

import openai
from convokit import Speaker, Utterance
from llama_cpp import Llama

logger = logging.getLogger(__name__)


def dprint(level="info", message=""):
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)


def getUtterances(message_list, speakers_list, disc_id, replyto_list=[]):
    if len(message_list) != len(speakers_list):
        raise ValueError("message list and speaker list do not have the same length.")
    speakers_unq = set(speakers_list)
    speakers = {speaker: Speaker(id=speaker) for speaker in speakers_unq}
    utterances = []
    counter = 0
    for utt, speaker in zip(message_list, speakers_list):
        utterances.append(
            Utterance(
                id=f"utt_{counter}_{disc_id}",
                speaker=speakers[speaker],
                conversation_id=str(disc_id),
                reply_to=replyto_list or None,
                text=utt,
            )
        )
        counter += 1
    return utterances, speakers


def save_dict_2_json(scores_dict, fine_name, conv_id, timestr):
    with open(f"{fine_name}_{conv_id}_{timestr}.json", "w", encoding="utf-8") as fout:
        json.dump(scores_dict, fout, ensure_ascii=False, indent=4)


def isValidResponse(parts):
    if len(parts) != 2:
        return -1
    value = parts[1]
    if len(value) > 10:
        match = re.match(r"^\s*(\d+)", value)
        if match:
            value = match.group(1)
            value = value.strip()
    rightParenthesisIndex = value.find("]")
    leftParenthesisIndex = value.find("[")
    if rightParenthesisIndex > 0 and leftParenthesisIndex > 0:
        value = value[leftParenthesisIndex:rightParenthesisIndex]
    value = value.replace("[", "").replace("]", "")
    if value.endswith("."):
        value = value.rstrip(".")
    try:
        value = float(value)
    except Exception as e:
        return -1
    if not isinstance(value, (int, float)):
        return -1
    return value


def extractFeature(feature, label):
    # if not label.startswith("-Label"):
    #    print("LLM output for utterance is ill-formatted, skipping utterance\n")
    #    return -1

    parts = label.split("\n")
    for j in parts:
        try:
            entries = j.split(":")

            key = entries[0]
            value = entries[1]

            key = key.replace("-", "")

            rightParenthesisIndex = value.find("]")
            leftParenthesisIndex = value.find("[")
            if rightParenthesisIndex > 0 and leftParenthesisIndex > 0:
                value = value[leftParenthesisIndex:rightParenthesisIndex]
            value = value.replace("[", "").replace("]", "")
            if value.endswith("."):
                value = value.rstrip(".")

            try:
                value = float(value)
            except Exception as e:
                value = -1
            feature[key] = value
        except Exception as e:
            print(e)
    return feature


def validateInputParams(model_type, openAIKEY, model_path):
    if model_type == "openai" and not openAIKEY:
        print("OpenAI API key does not exist. Scores will not be computed. Exiting")
        sys.exit(1)

    if model_type == "llama" and not model_path:
        print("Llama model path does not exist. Exiting")
        sys.exit(1)

    if model_type == "transformers" and not model_path:
        print("transformers model path does not exist. Exiting")
        sys.exit(1)

    if (
        model_type != "llama"
        and model_type != "openai"
        and model_type != "transformers"
    ):
        print("Expected model type: openai or llama or transformers. Exiting")
        sys.exit(1)

    return True


def sleep(model_type):
    pass
    # if model_type == "openai":
    # print("Sleeping for 60 seconds, for openAI quota")
    # time.sleep(60)


def prompt_gpt4(prompt, key, model_type, model):
    openai.api_key = key
    ok = False
    counter = 0
    while not ok:
        counter = counter + 1
        try:
            if model_type == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0,
                )
                result = response["choices"][0]["message"]["content"]
            elif model_type.lower() == "llama":
                messages = [{"role": "user", "content": prompt}]
                response = model.create_chat_completion(
                    messages=messages,
                    max_tokens=4096,
                )
                result = response["choices"][0]["message"]["content"]
                # result = llm(prompt, max_tokens=4096)
            elif model_type == "transformers":
                messages = [{"role": "user", "content": prompt}]
                response = model(
                    messages,
                    max_new_tokens=4096,
                    return_full_text=False,
                )[0]["generated_text"]
                result = response
            else:
                raise ValueError(
                    "Invalid model_type. Choose 'openai', 'llama' or 'transformers'."
                )
            ok = True
        except Exception as ex:
            print("error", ex)
            print("sleep for 5 seconds")
            time.sleep(5)
            if counter > 10:
                return -1
    return result


_cached_models = {}


def getModel(model_path, gpu, model_type="llama", device="auto"):
    key = (model_path, model_type, device, gpu)
    if key in _cached_models:
        dprint("info", " Using cached model for: {key}")
        return _cached_models[key]

    if model_type == "controversy":
        from transformers import pipeline

        pipe = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device_map=device,
        )
        _cached_models[key] = pipe
        return pipe

    if model_type == "echo":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model = "Qwen/Qwen1.5-4B-Chat"
        adapter_model = "Johndfm/ECoh-4B"
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, adapter_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = model.to(device)
        _cached_models[key] = (model, tokenizer)
        return (model, tokenizer)
    if model_type == "transformers":
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        generator = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )
        _cached_models[key] = generator
        return generator
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=50 if gpu else 0,
        n_ctx=4096 * 2,
    )
    _cached_models[key] = llm
    return llm

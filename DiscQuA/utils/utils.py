import json
import sys
import time

import openai
from convokit import Speaker, Utterance
from llama_cpp import Llama


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
    rightParenthesisIndex = value.find("]")
    leftParenthesisIndex = value.find("[")
    if rightParenthesisIndex > 0 and leftParenthesisIndex > 0:
        value = value[leftParenthesisIndex:rightParenthesisIndex]
    value = value.replace("[", "").replace("]", "").replace(".", "")
    if not isinstance(value, (int, float)):
        return -1
    return value


def validateInputParams(model_type, openAIKEY, model_path):
    if model_type == "openai" and not openAIKEY:
        print("OpenAI API key does not exist. Scores will not be computed. Exiting")
        sys.exit(1)

    if model_type == "llama" and not model_path:
        print("Llama model path does not exist. Exiting")
        sys.exit(1)

    if model_type != "llama" and model_type != "openai":
        print("Expected model type: openai or llama. Exiting")
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
            else:
                raise ValueError("Invalid model_type. Choose 'openai' or 'llama'.")
            ok = True
        except Exception as ex:
            print("error", ex)
            print("sleep for 5 seconds")
            time.sleep(5)
            if counter > 10:
                return -1
    return result


def getModel(model_path, gpu):
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=50 if gpu else 0,
        n_ctx=4096 * 2,
    )
    return llm

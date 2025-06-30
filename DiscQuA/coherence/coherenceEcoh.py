import time

from tqdm import tqdm

from DiscQuA.utils import save_dict_2_json

#############################################################
base_model = "Qwen/Qwen1.5-4B-Chat"
adapter_model = "Johndfm/ECoh-4B"


#############################################################
def inferenceModel(history, response, tokenizer, model, device):

    textForInference = f"Context:\n {history} \n\nResponse:\n {response}"

    messages = [
        {"role": "system", "content": "You are a Coherence evaluator."},
        {
            "role": "user",
            "content": f"{textForInference}\n\nGiven the context, is the response Coherent (Yes/No)? Explain your reasoning.",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def processDiscussion(message_list, speaker_list, disc_id, tokenizer, model, device):
    conversation_id = disc_id

    utterances = []

    counter = 0
    for utt, speaker in zip(message_list, speaker_list):
        text = utt.replace("\r\n", " ").replace("\n", " ").rstrip().lstrip()
        #
        utterances.append((text, speaker, f"conv_{disc_id}_utt_{counter}"))
        counter += 1
    #
    context_list = []
    response_list = []
    previous_context = ""
    seperator = " \n "
    context = ""
    for i, u in enumerate(utterances):
        text, speaker, utt_id = u
        if i + 1 >= len(utterances):
            break
        text = speaker + ": " + text
        if i == 0:
            context = text
            previous_context = text
            response_speaker = utterances[i + 1][1]
            response = response_speaker + ": " + utterances[i + 1][0]
        else:
            context = previous_context + seperator + " " + text
            response_speaker = utterances[i + 1][1]
            response = response_speaker + ": " + utterances[i + 1][0]
            previous_context = context
        context_list.append(context)
        response_list.append(response)

    inference_result_list = []
    for cont, resp in tqdm(zip(context_list, response_list), total=len(response_list)):
        result = inferenceModel(cont, resp, tokenizer, model, device)
        inference_result_list.append(result)
    return inference_result_list, context_list, response_list, conversation_id


def calculate_coherence_ecoh(message_list, speaker_list, disc_id, device="cpu"):
    """Calculates coherence scores for a discussion using a fine-tuned causal language model,
       returning binary judgments and rationales for each message.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speaker_list (list[str]): The corresponding list of speakers for each utterance.
        disc_id (str): Unique identifier for the discussion.
        device (str): The device to load the model on. If `None`, the device will be inferred. Defaults to cpu.

    Returns:
         dict[str, list[int]]: A mapping from the discussion ID to a list of binary labels,
        where 1 indicates a coherent response and 0 indicates incoherence, for each utterance.
    """

    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #############################################################
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    #############################################################
    model = model.to(device)
    #
    aggregate_results = {}
    aggregate_context = {}
    aggregate_response = {}
    #
    results, context_list, response_list, conversation_id = processDiscussion(
        message_list, speaker_list, disc_id, tokenizer, model, device
    )
    aggregate_results[conversation_id] = results
    aggregate_context[conversation_id] = context_list
    aggregate_response[conversation_id] = response_list

    booleans = {}
    reasons = {}
    for disc_id, value in aggregate_results.items():
        for res in value:
            parts = res.split("The answer is")
            reason = parts[0]
            res_str = parts[1]
            boolean_res = 0
            if "Yes" in res_str:
                boolean_res = 1
            if disc_id in booleans:
                booleans[disc_id].append(boolean_res)
                reasons[disc_id].append(reason)
            else:
                booleans[disc_id] = [boolean_res]
                reasons[disc_id] = [reason]
    #
    save_dict_2_json(
        aggregate_results, "discussion_turnlevel_agresults", disc_id, timestr
    )
    save_dict_2_json(reasons, "discussion_turnlevel_reasons", disc_id, timestr)
    save_dict_2_json(booleans, "discussion_turnlevel_boolean_results", disc_id, timestr)
    return booleans

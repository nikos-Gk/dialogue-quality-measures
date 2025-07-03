# -*- coding: utf-8 -*-
import time

from DiscQuA.utils import (
    getModel,
    getUtterances,
    isValidResponse,
    prompt_gpt4,
    save_dict_2_json,
    validateInputParams,
)

#################################################################################
prompt = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.
Rate the informativeness of the entire discussion on a scale of 1 to 5, where 1 is of poor information quality (uninformative) and 5 of high information quality (informative). 
Conclude your evaluation with the statement: 'The informativeness of the comments presented in the above discussion is: [X]', where X is the rating (integer) you've determined. 
Please, ensure that your last statement is the score in brackets [].
"""


def calculate_discussion_informativeness_score(utts, topic, key, model_type, model):
    conv_text = ""
    for utt in utts:
        text = utt.text
        speaker = utt.get_speaker().id
        conv_text = conv_text + speaker + ": " + text + "\n\n"
        formatted_prompt = prompt.format(conv_text=conv_text, post=topic)
    annotations_ci = []
    try:
        response_text = prompt_gpt4(formatted_prompt, key, model_type, model)
        # print(formatted_prompt)
        annotations_ci.append(response_text)
    except Exception as e:
        print("Error: ", e)
        annotations_ci.append(-1)
    return annotations_ci


def calculate_informativeness_conversation(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    device="auto"
):
    """Computes the overall informativeness score of a discussion using a large language model (LLM).

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        disc_id (str): Unique identifier for the discussion.
        openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
        model_type (str): Language model type to use, either "openai" or "llama" or "transformers". Defaults to "openai".
        model_path (str): Path to the model, used only for model_type "llama" or "transformers". Defaults to "".
        gpu (bool): A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.
        device(str): The device to load the model on. If None, the device will be inferred. Defaults to auto.

    Returns:
        dict[str, int]: A dictionary mapping the discussion ID to its computed overall informativeness score.
    """

    validateInputParams(model_type, openAIKEY, model_path)

    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    #
    inform_scores_llm_output_dict = {}
    utterances, speakers = getUtterances(
        message_list, speakers_list, disc_id, replyto_list=[]
    )
    conv_topic = message_list[0]

    print(
        "Overall Informativeness Score-Proccessing discussion: ",
        disc_id,
        " with LLM",
    )
    inform_score = calculate_discussion_informativeness_score(
        utterances, conv_topic, openAIKEY, model_type, llm
    )
    inform_scores_llm_output_dict[disc_id] = inform_score
    #
    save_dict_2_json(
        inform_scores_llm_output_dict, "llm_output_informativeness", disc_id, timestr
    )

    """        
    with open("llm_output_informativeness_.json", encoding="utf-8") as f:
        info_scores = json.load(f)
    inform_scores_llm_output_dict=info_scores
    """

    inform_scores_per_disc = {}
    for disc_id, turnAnnotations in inform_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall informativeness score , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split(
                "informativeness of the comments presented in the above discussion is:"
            )
            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing overall informativeness score , skipping discussion\n"
                )
                print(label)
                continue

            inform_scores_per_disc[disc_id] = value
    #
    save_dict_2_json(
        inform_scores_per_disc, "inform_scores_per_discussion", disc_id, timestr
    )
    return inform_scores_per_disc

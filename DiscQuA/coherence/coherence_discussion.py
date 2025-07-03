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

prompt = """{conv_text}'\n\n\n

The texts above show a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating inflammatory and aggressive speech.
You are a coherence evaluator. Rate the coherence of the entire discussion on a scale from 1 to 5, where 1 is of poor coherence quality (incoherent) and 5 of high coherence quality (coherent). 
Conclude your evaluation with the statement: 'The coherence of the comments presented in the above discussion is: [X]', where X is the numeric score (integer) you've determined. 
Please, provide the final answer directly with no reasoning steps.
Please, ensure that your last statement is the score in brackets [].

"""


def calculate_discussion_coherence_score(utts, topic, key, model_type, model):
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


def calculate_coherence_conversation(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    device="auto"
):
    """Evaluates the overall coherence of a discussion on a scale from 1 to 5, where 1 is of poor coherence quality (incoherent) and 5 of high coherence quality (coherent).

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
         dict: A dictionary containing the final coherence score (integer) for the given discussion.
    """

    validateInputParams(model_type, openAIKEY, model_path)

    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    coherence_scores_llm_output_dict = {}
    #
    utterances, speakers = getUtterances(
        message_list, speakers_list, disc_id, replyto_list=[]
    )
    conv_topic = message_list[0]
    print("Overall Coherence Score-Proccessing discussion: ", disc_id, " with LLM")
    coh_score = calculate_discussion_coherence_score(
        utterances, conv_topic, openAIKEY, model_type, llm
    )
    coherence_scores_llm_output_dict[disc_id] = coh_score
    #
    save_dict_2_json(
        coherence_scores_llm_output_dict,
        "llm_output_coherence",
        disc_id,
        timestr,
    )

    """               
        with open("llm_output_coherence_.json", encoding="utf-8") as f:
            coh_scores = json.load(f)
        coherence_scores_llm_output_dict=coh_scores
    """

    coherence_scores_per_disc = {}
    for disc_id, turnAnnotations in coherence_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall coherence score , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split(
                "coherence of the comments presented in the above discussion is:"
            )

            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing overall coherence score , skipping discussion\n"
                )
                print(label)
                continue
            coherence_scores_per_disc[disc_id] = value

    save_dict_2_json(
        coherence_scores_per_disc, "coherence_scores_per_disc", disc_id, timestr
    )
    return coherence_scores_per_disc

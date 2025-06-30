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
Now, please use chain-of-thought reasoning to rate the diversity of the arguments of the above discussion.
Note that a high number of diverse arguments is indicative of the deliberativeness of the discussion.
Also, can enhance the (argument) quality of discussions by bringing varied perspectives, backgrounds, and experiences into the discussion.
After the Chain-of-Thoughts reasoning steps, rate the diversity of the arguments of the entire discussion on a scale from 1 to 5, where 1 is of poor diversity quality (arguments of high similarity) and 5 of high diversity quality (diverse arguments). 
Conclude your evaluation with the statement: 'The diversity of the arguments of the above discussion is: [X]', where X is the rating you've determined. 
Please, ensure that your last statement is the score in brackets [].
"""


def calculate_discussion_diversity_score(utts, topic, key, model_type, model):
    conv_text = ""
    for utt in utts:
        text = utt.text
        speaker = utt.get_speaker().id
        conv_text = conv_text + speaker + ": " + text + "\n\n"
        formatted_prompt = prompt.format(conv_text=conv_text, post=topic)
    annotations_ci = []
    try:
        # response_text = prompt_gpt4(formatted_prompt, key, model_type, model)
        print(formatted_prompt)
        # annotations_ci.append(response_text)
    except Exception as e:
        print("Error: ", e)
        annotations_ci.append(-1)
    return annotations_ci


def calculate_diversity_conversation(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
):
    """Assings an overall conversational diversity score for the given discussion based on linguistic variety,
    argumentation patterns, and idea breadth. The evaluation is performed using a large language model
    (OpenAI or LLaMA).
    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        disc_id (str): Unique identifier for the discussion.
        openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
        model_type (str): Language model type to use, either "openai" or "llama". Defaults to "openai".
        model_path (str): Path to the local LlaMA model directory, used only if model_type is "llama". Defaults to "".
        gpu (bool):  A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.

    Returns:
         dict[str, float]: Dictionary mapping the discussion ID to its overall LLM-assigned diversity score.
    """

    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None
    if model_type == "llama":
        llm = getModel(model_path, gpu)
    #
    divers_scores_llm_output_dict = {}
    #
    utterances, speakers = getUtterances(
        message_list, speakers_list, disc_id, replyto_list=[]
    )
    conv_topic = message_list[0]
    print("Overall Diversity Score-Proccessing discussion: ", disc_id, " with LLM")
    div_score = calculate_discussion_diversity_score(
        utterances, conv_topic, openAIKEY, model_type, llm
    )
    divers_scores_llm_output_dict[disc_id] = div_score
    #
    save_dict_2_json(
        divers_scores_llm_output_dict, "llm_output_diversity", disc_id, timestr
    )

    """         
    with open("llm_output_diversity_.json", encoding="utf-8") as f:
        div_scores = json.load(f)
    divers_scores_llm_output_dict=div_scores
    """

    div_scores_per_disc = {}
    for disc_id, turnAnnotations in divers_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall diversity score , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split(
                "diversity of the arguments of the above discussion is:"
            )
            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing overall diversity score , skipping discussion\n"
                )
                print(label)
                continue

            div_scores_per_disc[disc_id] = value

    save_dict_2_json(
        div_scores_per_disc, "diversity_scores_per_discussion", disc_id, timestr
    )
    return div_scores_per_disc

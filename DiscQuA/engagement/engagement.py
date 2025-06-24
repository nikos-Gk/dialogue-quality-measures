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
Now, please use chain-of-thought reasoning to rate the engagement of the above discussion.
Engagement refers to the degree to which a discussion sustains interest and participation.
After the Chain-of-Thoughts reasoning steps, rate the engagement of the entire discussion on a scale of 1 to 5, where 1 is of poor engagement quality (that is the comments do not sustain interest and participation) and 5 of high engagement quality (comments sustain interest and participation to a high degree). 
Conclude your evaluation with the statement: 'The engagement quality of the above discussion is: [X]', where X is the rating you've determined. 
Please, ensure that your last statement is the score in brackets [].
"""


def calculate_discussion_engagement_score(utts, topic, key, model_type, model):
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


def engagement(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
):
    validateInputParams(model_type, openAIKEY, model_path)

    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None
    if model_type == "llama":
        llm = getModel(model_path, gpu)

    engag_scores_llm_output_dict = {}
    utterances, speakers = getUtterances(
        message_list, speakers_list, disc_id, replyto_list=[]
    )
    #
    conv_topic = message_list[0]
    #
    print("Overall Engagement Score-Proccessing discussion: ", disc_id, " with LLM")
    engag_score = calculate_discussion_engagement_score(
        utterances, conv_topic, openAIKEY, model_type, llm
    )
    engag_scores_llm_output_dict[disc_id] = engag_score
    #
    save_dict_2_json(
        engag_scores_llm_output_dict, "llm_output_engagement", disc_id, timestr
    )

    """        
    with open("llm_output_engagement_.json", encoding="utf-8") as f:
        engag_scores = json.load(f)
    engag_scores_llm_output_dict=engag_scores
    """

    engag_scores_per_disc = {}
    for disc_id, turnAnnotations in engag_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing overall engagement score , skipping discussion\n"
                )
                print(label)
                continue
            parts = label.split("engagement quality of the above discussion is:")
            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing overall engagement score , skipping discussion\n"
                )
                print(label)
                continue

            engag_scores_per_disc[disc_id] = value

    save_dict_2_json(
        engag_scores_per_disc, "engag_scores_per_discussion", disc_id, timestr
    )
    return engag_scores_per_disc

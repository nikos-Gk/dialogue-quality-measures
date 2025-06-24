# -*- coding: utf-8 -*-
import json
import sys
import time

import pandas as pd
from convokit import Speaker, Utterance

from DiscQuA.utils import (
    getModel,
    getUtterances,
    isValidResponse,
    prompt_gpt4,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

prompt = """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a 
discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating 
inflammatory and aggressive speech.

*CONVERSATION HISTORY*: "{conv_history}"

*RESPONSE*: "{response}"

You are a coherence evaluator. 
Given the post that the discussion is based on and the conversation history, you have to assign a score on a scale of 1 to 5 that indicates the coherence of the response.
A score 1 indicates that the response is of poor coherence quality (incoherent), while 5 indicates that the response is of extremely high coherence quality (coherent).
Noteworthy, the conversation history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the coherence of the new response.
Please provide the final answer directly with no reasoning steps.
For clarity, your evaluation should be presented with the statement: 'The coherence of the new response is: [X]', where X is the numeric score (integer number) you've determined. 
Please, ensure that your last statement is the score in brackets [].
"""


def calculate_response_coherence_score(utts, topic, openAIKEY, model_type, model, ctx):
    annotations_ci = []
    for index, utt in enumerate(utts):
        conv_hist = ""
        text = utt.text
        speaker = utt.get_speaker().id

        if index > 0:
            start_ctx = max(0, index - ctx)
            for i in range(start_ctx, index):
                prev_utt = utts[i]
                prev_speaker = prev_utt.get_speaker().id
                prev_text = prev_utt.text
                conv_hist += f"\n<user_name={prev_speaker}>\n{prev_text}\n"

        try:
            formatted_prompt = prompt.format(
                response="<user_name=" + speaker + ">" + "\n" + text,
                conv_history=conv_hist,
                post=topic,
            )
            response_text = prompt_gpt4(formatted_prompt, openAIKEY, model_type, model)
            # print(formatted_prompt)
            annotations_ci.append(response_text)
        except Exception as e:
            print("Error: ", e)
            annotations_ci.append(-1)
    return annotations_ci


def calculate_coherence_response(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    ctx=1,
):
    validateInputParams(model_type, openAIKEY, model_path)

    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    llm = None
    if model_type == "llama":
        llm = getModel(model_path, gpu)

    coh_per_resp_scores_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        print(
            "Coherence Score Per Response-Proccessing discussion: ",
            disc_id,
            " with LLM",
        )
        coh_per_resp = calculate_response_coherence_score(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        coh_per_resp_scores_llm_output_dict[disc_id] = coh_per_resp
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        coh_per_resp_scores_llm_output_dict,
        "llm_output_coh_per_response",
        disc_id,
        timestr,
    )

    """            
        with open("llm_output_coh_per_response_.json", encoding="utf-8") as f:
            coh_scores = json.load(f)
        coh_per_resp_scores_llm_output_dict=coh_scores
    """

    coherence_scores_per_response = {}
    for disc_id, turnAnnotations in coh_per_resp_scores_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing coherence response score , skipping response\n"
                )
                print(label)
                counter += 1
                continue
            parts = label.split("coherence of the new response is:")

            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing coherence response score , skipping discussion\n"
                )
                print(label)
                counter += 1
                continue

            ut_dict[disc_id + "_" + str(counter)] = value
            counter += 1
        coherence_scores_per_response[disc_id] = ut_dict

    save_dict_2_json(
        coherence_scores_per_response, "coherence_per_response", disc_id, timestr
    )

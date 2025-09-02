# -*- coding: utf-8 -*-
import time

from tqdm import tqdm

from DiscQuA.utils import (
    dprint,
    extractFeature,
    getModel,
    getUtterances,
    prompt_gpt4,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

ini = """Below is a set of labels from a social bias frame (presented in Sap et al., 2019), properly designed for modelling the pragmatic frames in which people project social biases and stereotypes onto others."""
#################################################################################
social_bias_labels = """ Social bias Labels:

Label 0: Offensiveness - Comment is considered offensive, disrespectful, or toxic to anyone/someone.

Label 1: Intent to offend - The intent of the comment is to be offensive/disrespectful to anyone.

Label 2: Lewd - The comment contain or allude to sexual content/acts.

Label 3: Group implications - The comment imply offense/disrespect to an identity-related group of people (e.g., ''f... muslims'', ''f... you, you s... nigger'') and not just an insult to an individual or non-identity-related group of people (e.g., ''f... you'', ''All y'all are m...'').

Label 4: In-group language - The comment imply offense/disrespect to an identity-related group of people (e.g., ''f... muslims'', ''f... you, you s... nigger'') and not just an insult to an individual or non-identity-related group of people (e.g., ''f... you'', ''All y'all are m...'') and the author of the comment sound like they belong to the same social/demographic group that is targeted.                                                                                                                                                                                                                                 

Label 5: Other - For comments not covered by the above labels.

"""
#################################################################################
final = """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
Each utterance is a brief Reddit-based comment responding to the post and other users' comments on it.

*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

You are a social bias evaluator.

Noteworthy, the conversation history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the new utterance.

Thus, please do not annotate the entire conversation but annotate only the new utterance by determining its appropriate social bias label (s). Please provide the final answer directly with no reasoning steps.

If the new utterance fits multiple categories, list all that apply. Ensure that your final answer clearly identifies whether each of the social bias labels (0-5) are applicable (1) or not applicable (0) to the new utterance.

For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of each label as follows:

- Label 0: [1/0]
- Lable 1: [1/0]
- Lable 2: [1/0]
- Label 3: [1/0]
- Label 4: [1/0]
- Label 5: [1/0]
"""


def calculate_social_bias_labels(utts, topic, openAIKEY, model_type, model, ctx):

    prompt = ini + social_bias_labels + final
    annotations_ci = []
    for index, utt in enumerate(tqdm(utts, desc="Processing utterances")):
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
                utterance="<user_name=" + speaker + ">" + "\n" + text,
                conv_history=conv_hist,
                post=topic,
            )
            # print(formatted_prompt)
            response_text = prompt_gpt4(formatted_prompt, openAIKEY, model_type, model)
            annotations_ci.append(response_text)
        except Exception as e:
            print("Error: ", e)
            annotations_ci.append(-1)
    return annotations_ci


def calculate_social_bias(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    ctx=1,
    device="auto",
):
    """Analyzes social bias in a discussion using a language model (OpenAI or LLaMA) and returns bias annotations per utterance or for the entire discussion.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        disc_id (str): Unique identifier for the discussion.
        openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
        model_type (str): Language model type to use, either "openai" or "llama" or "transformers". Defaults to "openai".
        model_path (str): Path to the model, used only for model_type "llama" or "transformers". Defaults to "".
        gpu (bool): A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.
        ctx (int): A boolean flag; if True, the annotations are applied at the discussion level; otherwise at the utterance level.
        device(str): The device to load the model on. If None, the device will be inferred. Defaults to auto.

    Returns:
        dict: Dictionary containing social bias annotations per utterance for the given discussion.
    """
    validateInputParams(model_type, openAIKEY, model_path)
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)
    #
    socialbiaslabels_llm_output_dict = {}
    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        dprint(
            "info", f"Social bias labels-Proccessing discussion: {disc_id} with LLM "
        )
        #
        socialbias_features = calculate_social_bias_labels(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        socialbiaslabels_llm_output_dict[disc_id] = socialbias_features
        #
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        socialbiaslabels_llm_output_dict,
        "llm_output_socialbias_per_response",
        disc_id,
        timestr,
    )

    """
    with open("llm_output_socialbias_per_response_.json", encoding="utf-8") as f:
        sb_labels = json.load(f)
    socialbiaslabels_llm_output_dict=sb_labels
    """

    social_bias_per_response = {}
    for disc_id, turnAnnotations in socialbiaslabels_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                dprint(
                    "info",
                    "LLM output with missing social bias label , skipping response\n",
                )
                dprint("info", label)
                counter += 1
                continue

            feature = {}
            feature = extractFeature(feature, label)
            if feature == -1:
                counter += 1
                continue

            key_iter = "utt_" + str(counter)
            ut_dict[key_iter] = [feature]
            counter += 1

        if disc_id in social_bias_per_response:
            social_bias_per_response[disc_id].append(ut_dict)
        else:
            social_bias_per_response[disc_id] = [ut_dict]

    save_dict_2_json(
        social_bias_per_response, "socialbias_per_utterance", disc_id, timestr
    )
    return social_bias_per_response

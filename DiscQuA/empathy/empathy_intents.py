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

####################################################################################################################################################################################################################################################################################################
ini = """Below is a dialogue act taxonomy (presented in Welivita and Pu, 2020), designed to capture the underlying intent of utterances in a discussion in terms of conveying empathy. Empathy refers to the ability to understand others’ perspectives and emotions and respond correspondingly."""
####################################################################################################################################################################################################################################################################################################
expressed_empathy_labels = """\n\n

Label 0: Questioning - A comment that asks for clarifications or further details.

Label 1: Acknowledging - A comment that expresses recognition or validation of another participant’s feelings, experiences, or viewpoint, without necessarily indicating agreement.

Label 2: Agreeing - A comment that expresses alignment or support for another participant’s viewpoint, statement, or emotion.

Label 3: Consoling -A comment that consoles another participant by offering comfort, reassurance, or emotional support in response to something upsetting or difficult they’ve shared (negative emotional context). 

Label 4: Encouraging – A comment that encourages another participant by offering support, affirmation, or positive reinforcement in response to something hopeful, exciting, or uplifting they’ve shared (positive emotional context).
    
Label 5: Sympathizing - A comment that sympathizes with another participant by expressing pity, sorrow, or shared emotional concern in response to something distressing or unfortunate they’ve shared (negative emotional context).

Label 6: Wishing – A comment that expresses hope, goodwill, or positive intentions toward another participant, often in response to something they’ve shared or as a gesture of kindness.

Label 7: Suggesting – A comment that offers advice, recommendations, or possible actions to another participant, often in response to a concern, question, or situation they’ve shared.

Label 8: Sharing own thoughts/opinion – A comment in which a participant expresses their personal views, beliefs, or reflections, often contributing to the discussion without directly responding to another participant’s input.

Label 9: Sharing or relating to own experience – A comment in which a participant recounts a personal experience or relates their own story, often to connect with or illustrate a point in the discussion.

Label 10: Advising - A comment that provides guidance, recommendations, or suggestions.

Label 11: Expressing care or concern - Comment that expresses care or concern toward another participant.

Label 12: Expressing relief - A comment that conveys relief directed at another participant, often following a situation that was stressful, uncertain, or emotionally intense for them.

Label 13: Disapproving - A comment that expresses disagreement, criticism, or disapproval toward another participant’s viewpoint, actions, or statements. It often challenges a prior claim and conveys a contrasting opinion, concern, or negative judgment.

Label 14: Appreciating - A comment that expresses positive recognition, gratitude, or admiration toward another participant.
"""


final = """\n\n
You are an annotator of expressed empathetic intentions.
Given a discussion history, please analyze a new utterance from a user in a conversation discussing with others about this potentially controversial post.
Post: {post} 
    
*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

Noteworthy, the discussion history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the new utterance.

Thus, please do not annotate the entire discussion but annotate only the new utterance by determining its appropriate expressed empathy label(s). Please provide the final answer directly with no reasoning steps.

If the new utterance fits multiple categories, list all that apply. Ensure that your final answer clearly identifies whether each of the expressed empathy labels (0-14) are applicable (1) or not applicable (0) to the new utterance.

For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of label as follows:

- Label 0: [1/0]
- Label 1: [1/0]
- Label 2: [1/0]
- Label 3: [1/0]
- Label 4: [1/0]
- Label 5: [1/0]
- Label 6: [1/0]
- Label 7: [1/0]
- Label 8: [1/0]
- Label 9: [1/0]
- Label 10: [1/0]
- Label 11: [1/0]
- Label 12: [1/0]
- Label 13: [1/0]
- Label 14: [1/0]
"""


def calculate_expressed_empathy_labels(utts, topic, openAIKEY, model_type, model, ctx):
    prompt = ini + expressed_empathy_labels + final
    annotations_ci = []
    #
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
            response_text = prompt_gpt4(formatted_prompt, openAIKEY, model_type, model)
            # print(formatted_prompt)
            annotations_ci.append(response_text)

        except Exception as e:
            print("Error: ", e)
            annotations_ci.append(-1)
    return annotations_ci


def expressed_empathy(
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
    """Assigns expressed empathy labels to utterances in a discussion using a large language model (LLM).
    Labels reflect the true intention of each utterance in terms of conveying empathy.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        disc_id (str): Unique identifier for the discussion.
        openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
        model_type (str): Language model type to use, either "openai" or "llama" or "transformers". Defaults to "openai".
        model_path (str): Path to the model, used only for model_type "llama" or "transformers". Defaults to "".
        gpu (bool): A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.
        ctx (int): Number of previous utterances to include as context for each input. Defaults to 1.
        device(str): The device to load the model on. If None, the device will be inferred. Defaults to auto.

    Returns:

        dict[str, dict[str, dict[str, int]]]: Dictionary mapping the discussion ID to a dictionary
        of utterance-level empathy labels. Each inner dictionary maps utterance IDs (e.g., "utt_0")
        to a set of extracted expressed empathy features as key-value pairs.

    """

    validateInputParams(model_type, openAIKEY, model_path)
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    expressed_empathy_labels_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        dprint(
            "info",
            f"Expressed empathy labels Per Response-Proccessing discussion: {disc_id} with LLM ",
        )

        empathy_labels_per_resp = calculate_expressed_empathy_labels(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        expressed_empathy_labels_llm_output_dict[disc_id] = empathy_labels_per_resp
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        expressed_empathy_labels_llm_output_dict,
        "llm_output_expr_empathy_labels_per_response",
        disc_id,
        timestr,
    )

    """        
    with open("llm_output_expr_empathy_labels_per_response_.json", encoding="utf-8") as f:
        expr_empathy_labels = json.load(f)
    expressed_empathy_labels_llm_output_dict=expr_empathy_labels
    """
    expr_empathy_labels_per_response = {}
    for disc_id, turnAnnotations in expressed_empathy_labels_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                dprint(
                    "info",
                    "LLM output with missing expressed empathy label, skipping utterance\n",
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
            ut_dict[key_iter] = feature
            counter += 1

        expr_empathy_labels_per_response[disc_id] = ut_dict

    save_dict_2_json(
        expr_empathy_labels_per_response,
        "expressed_empathy_label_per_response",
        disc_id,
        timestr,
    )
    return expr_empathy_labels_per_response

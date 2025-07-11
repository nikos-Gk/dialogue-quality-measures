# -*- coding: utf-8 -*-
import time
from tqdm import tqdm


from DiscQuA.utils import (
    getModel,
    getUtterances,
    isValidResponse,
    prompt_gpt4,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

prompt = """
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating inflammatory and aggressive speech.

*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

You are a toxicity evaluator. 
Given the post that the discussion is based on and the conversation history, you have to assign a label that indicates the toxicity of the response.
The possible labels are 1: Not Toxic, 2: Barely Toxic, 3: Almost Toxic, 4: Toxic and 5: Extremely Toxic.
Noteworthy, the conversation history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the toxicity of the new utterance.
Please provide the final answer directly with no reasoning steps.
For clarity, your evaluation should be presented with the statement: 'The toxicity of the new utterance is: [X]', where X is the label (integer) you've determined. 
Please, ensure that your last statement is the label in brackets [].
"""


def calculate_response_toxicity_score(utts, topic, openAIKEY, model_type, model, ctx):
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


def calculate_toxicity(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    ctx=1,
    device="auto"
):
    """Evaluates the toxicity level of each utterance in a discussion using a selected language model, with optional context from previous utterances.

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
        dict: A dictionary mapping each utterance ID to its associated toxicity label, structured per discussion ID.
    """

    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    tox_per_resp_scores_llm_output_dict = {}
    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]

        print(
            "Toxicilty Label Per Response-Proccessing discussion: ",
            disc_id,
            " with LLM",
        )
        #
        tox_per_resp = calculate_response_toxicity_score(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        tox_per_resp_scores_llm_output_dict[disc_id] = tox_per_resp
        #
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        tox_per_resp_scores_llm_output_dict,
        "llm_output_tox_per_response",
        disc_id,
        timestr,
    )

    """            
    with open("llm_output_tox_per_response_.json", encoding="utf-8") as f:
        tox_scores = json.load(f)
    tox_per_resp_scores_llm_output_dict=tox_scores
    """

    toxicity_scores_per_response = {}
    for disc_id, turnAnnotations in tox_per_resp_scores_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                print(
                    "LLM output with missing toxicity response label , skipping response\n"
                )
                print(label)
                counter += 1
                continue
            parts = label.split("toxicity of the new utterance is:")
            value = isValidResponse(parts)
            if value == -1:
                print(
                    "LLM output with missing toxicity response label , skipping response\n"
                )
                print(label)
                counter += 1
                continue

            ut_dict["utt_" + str(counter)] = value
            counter += 1
        toxicity_scores_per_response[disc_id] = ut_dict

    save_dict_2_json(
        toxicity_scores_per_response, "toxicity_per_response", disc_id, timestr
    )
    return toxicity_scores_per_response

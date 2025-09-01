import time

from tqdm import tqdm

from DiscQuA.utils import (
    extractFeature,
    getModel,
    getUtterances,
    prompt_gpt4,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

ini = """Below is a set of labels used to categorize the overall emotional tone of a commnent within a discussion, as presented by Zhou et al. (2024)."""
#################################################################################
sentiment_labels = """ Sentiment labels:
Label 0: Negative - Comment that conveys an overall negative emotional tone, such as criticism, frustration, anger, or dissatisfaction (e.g., That’s not good use of inheritance).
Label 1: Neutral - Comment that expresses no strong emotional tone, often factual, objective, or emotionally balanced without leaning positive or negative (e.g., Are we planning on making use of this other places?).
Label 2: Positive - Comment that conveys an overall positive emotional tone, such as appreciation, enthusiasm, agreement, or encouragement (e.g., It looks slightly magical).
"""
#################################################################################
final = """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
Each utterance is a brief Reddit-based comment responding to the post and other users' comments on it.
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating  inflammatory and aggressive speech.
*CONVERSATION HISTORY*: "{conv_history}"
*NEW UTTERANCE*: "{utterance}"
You are an overall emotional tone annotator.
Noteworthy, the conversation history is provided for you to simply understand the utterance(s) made before the new utterance so as to help you better annotate the new utterance.
Thus, please do not annotate the entire conversation but annotate only the new utterance by determining its appropriate overall emotional tone. Please provide the final answer directly with no reasoning steps.
Ensure that your final answer clearly identifies whether each of the sentiment labels (0-2) are applicable (1) or not applicable (0) to the new utterance.
For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of each label as follows:

- Label 0: [1/0]
- Lable 1: [1/0]
- Lable 2: [1/0]
"""


def calculate_sentiment_labels(utts, topic, openAIKEY, model_type, model, ctx):

    prompt = ini + sentiment_labels + final
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


def sentiment_analysis(
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
    """Annotates the utterances in a discussion using the sentiment labels presented by Zhou et al. (2024).
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
        dict: A dictionary mapping discussion IDs to lists of per-utterance sentiment labels dictionaries.
              Each entry is keyed by utterance ID (e.g., "utt_0") and contains extracted features.

    """

    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None
    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    #
    sentimentlabels_llm_output_dict = {}
    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        print("Sentiment labels-Proccessing discussion: ", disc_id, " with LLM")
        #
        sentiment_features = calculate_sentiment_labels(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        sentimentlabels_llm_output_dict[disc_id] = sentiment_features
        #
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        sentimentlabels_llm_output_dict,
        "llm_output_sentimentlabel_per_response",
        disc_id,
        timestr,
    )

    """
    with open("llm_output_sentimentlabel_per_response_.json", encoding="utf-8") as f:
        sentiment_labels = json.load(f)
    sentimentlabels_llm_output_dict=sentiment_labels
    """

    sentimentlabel_per_response = {}
    for disc_id, turnAnnotations in sentimentlabels_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                print("LLM output with missing sentiment label, skipping response\n")
                print(label)
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

        if disc_id in sentimentlabel_per_response:
            sentimentlabel_per_response[disc_id].append(ut_dict)
        else:
            sentimentlabel_per_response[disc_id] = [ut_dict]

    save_dict_2_json(
        sentimentlabel_per_response, "sentimentlabel_per_utterance", disc_id, timestr
    )
    return sentimentlabel_per_response

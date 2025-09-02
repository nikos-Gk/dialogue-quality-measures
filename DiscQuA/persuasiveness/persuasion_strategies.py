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

ini = """Below is a set of persuasion strategies presented by Chen et al. (2025). 
Note that the role of a persuasion strategy is to influence or change the perceptions, opinions, attitudes, or behaviors of persuadees from a psychological standpoint."""
#################################################################################
persuassion_strategies = """ Persuassion strategies:
Label 0: Present of facts- Comment that uses factual evidence (e.g., official news reports, statistics) and a credible reasoning process to persuade others.
Label 1: Challenges and inquiries - Comment that expresses disbelief or opposition to another user's viewpoint and provides strong rebuttal evidence to enhance persuasiveness.
Label 2: Emotion eliciting - Comment that elicits specific emotions (e.g., anger, guilt) to influence others’ attitudes .
Label 3: Self-modeling - Comment in which the user shares their own intended actions, behaviors, or choices as a way to lead by example and encourage others to follow.
Label 4: Building trust - Comment that fosters psychological trust and mutual understanding by engaging in respectful, harmonious conversation.
Label 5: Courtesy tips - Comment that expresses gratitude, approval, praise to create a sense of goodwill and ease psychological tension.
Label 6: Compromise - Comment that expresses concessions on time to avoid making the discussion too intense causing someone to leave the discussion.
Label 7: Attachment of views - Comment that expresses kindness and concern through active listening and to some extent seconding another user's point of view.
Label 8: Problem decomposition - Comment that breaks down a complex persuasive goal into smaller, manageable sub-issues—often guiding the conversation step-by-step to clarify positions, reduce misunderstandings, and build consensus across multiple contributors.
Label 9: Social identity - Comment that emphasizes shared group membership, values, or identity—such as political affiliation, cultural background, professional role, or community ties—to build psychological support and strengthen persuasive impact in a multi-speaker online setting.
Label 10: Herd mentality  - Comment that leverages the perceived majority opinion, widespread social norms, or trending viewpoints to persuade others—suggesting that acceptance by many implies validity or correctness in a multi-speaker online setting.
Label 11: Emotive framing- Comment that conveys a strong personal stance, feeling, or emotional tone to amplify the persuasive impact and resonate with others in a multi-speaker online setting.
Label 12: Logical appeal - Comment that enhances the the credibility of persuasive content through the logical and reasoning process.
Label 13: Task Inquiry - Comment that asks questions related to persuasive goals.
Label 14: Personal story - Comment that shares a personal experience or narrative to illustrate the positive outcomes of one’s actions—used to inspire others, build relatability, and strengthen the persuasive appeal within a multi-speaker online conversation.
Label 15: Refutation of objections - Comment that directly challenges another participant’s viewpoint by presenting counterarguments, evidence, or reasoning.
"""
#################################################################################
final = """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
Each utterance is a brief Reddit-based comment responding to the post and other users' comments on it.
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating  inflammatory and aggressive speech.
*CONVERSATION HISTORY*: "{conv_history}"
*NEW UTTERANCE*: "{utterance}"
You are a persuassion strategy annotator.
Noteworthy, the conversation history is provided for you to simply understand the utterance(s) made before the new utterance so as to help you better annotate the new utterance.
Thus, please do not annotate the entire conversation but annotate only the new utterance by determining its appropriate persuassion strategy (-ies). Please provide the final answer directly with no reasoning steps.
If the new utterance fits multiple strategies, list all that apply. Ensure that your final answer clearly identifies whether each of the persuassion strategy (0-15) are applicable (1) or not applicable (0) to the new utterance.
For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of each strategy:

- Label 0: [1/0]
- Lable 1: [1/0]
- Lable 2: [1/0]
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
- Label 15: [1/0]
"""


def persuassion_steategy(utts, topic, openAIKEY, model_type, model, ctx):
    prompt = ini + persuassion_strategies + final
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


def calculate_persuasion_strategy(
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
    """Annotates the utterances in a discussion using persuasion strategies based on the taxonomy presented by Chen et al. (2025).
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
        dict: A dictionary mapping discussion IDs to lists of per-utterance persuasion strategies dictionaries.
              Each entry is keyed by utterance ID (e.g., "utt_0") and contains extracted persuasion strategies.

    """

    validateInputParams(model_type, openAIKEY, model_path)
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None
    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    #
    pers_strategies_llm_output_dict = {}
    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        dprint(
            "info", f"Persuasion strategy-Proccessing discussion: {disc_id} with LLM "
        )
        #
        pers_strategies = persuassion_steategy(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        pers_strategies_llm_output_dict[disc_id] = pers_strategies
        #
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        pers_strategies_llm_output_dict,
        "llm_output_persstrategy_per_response",
        disc_id,
        timestr,
    )

    """
    with open("llm_output_persstrategy_per_response_.json", encoding="utf-8") as f:
        pers_strategies = json.load(f)
    pers_strategies_llm_output_dict=pers_strategies
    """
    persstrat_per_response = {}
    for disc_id, turnAnnotations in pers_strategies_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                dprint(
                    "info",
                    "LLM output with missing persuasion strategy, skipping response\n",
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

        if disc_id in persstrat_per_response:
            persstrat_per_response[disc_id].append(ut_dict)
        else:
            persstrat_per_response[disc_id] = [ut_dict]

    save_dict_2_json(
        persstrat_per_response, "persstrategy_per_utterance", disc_id, timestr
    )
    return persstrat_per_response

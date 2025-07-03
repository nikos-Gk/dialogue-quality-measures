import time
from tqdm import tqdm


from DiscQuA.utils import (
    getModel,
    getUtterances,
    prompt_gpt4,
    save_dict_2_json,
    sleep,
    validateInputParams,
    extractFeature,
    isValidResponse
)

ini = """Below is a set of labels used to code individual speech acts across 14 independent indicators of deliberative quality (as presented by Fournier-Tombs and MacKenzie, 2021; Zhang, Culbertson and Paritosh, 2017).
Note that the role of a speech act is to indicate the intent of each utterance in a discussion."""
#################################################################################
speech_act_labels = """ Speech acts labels:
Label 0: Interruption - Comment that interrupts another (previous) comment.
Label 1: Explanation - Comment that provides a minimum level of context for the claims or opinions that are expressed.
Label 2: Causal reasoning - Comment that makes explicit causal connections between any observations, values, or objectives and the claims, conclusions, or recommendations that are made.
Label 3: Narrative - Comment that employs personall story telling to justify claims or values.
Label 4: Question - Comment that asks for clarifications or input.
Label 5: Response - Comment that replies to a question directed toward it.
Label 6: Advocacy - Comment that explicitly defends or advances the interests or claims of identifiable groups or communities.
Label 7: Public interest - Comment that attempts to connect claims, policies, or recommendations to the interests of the community as a whole.
Label 8: Disrespect - Comment that contains insults, dispersions, misrepresentations, name calling, and dismissive or disrespectful statements.
Label 9: Respect - Comment that contains explicit shows of respect, such as salutations, complements, or apologies.
Label 10: Counterarguments - Comment that engages with critiques made by other comments or attempts to address or respond to counter claims, concerns, or countervailing evidence.
Label 11: Constructive proposal - Comment that propose solutions to shared problems, alternative options, or compromises.
Label 12: Sarcasm (mocking)- Comment that is primarily a joke, a piece of sarcasm, or a pun intended to get a laugh or be silly but not trying to add information. Note that if the comment is sarcastic but using sarcasm to make a point or provide feedback, then the comment should not be be classified under this label.
Label 13: Sarcasm (nomocking)- Comment that uses sarcasm in a non-mocking way (e.g., aimed at devaluing the reference object) but just as a communication function that aligns with the nature of the point that is being made, or necessary to communicate to a particular audience.
"""
#################################################################################
final = """\n\n
You will be presented with a conversation history (which can be empty if the new utterance is the first utterance made in the conversation) from a discussion in an online chatroom with respect to this potentially controversial post between two or more individuals.
Post: {post}
Each utterance is a brief Reddit-based comment responding to the post and other users' comments on it.
All individuals answer to each other by presenting arguments on why they think the post(s) is or isn't reasonable, possibly incorporating  inflammatory and aggressive speech.
*CONVERSATION HISTORY*: "{conv_history}"
*NEW UTTERANCE*: "{utterance}"
You are a speech act annotator.
Noteworthy, the conversation history is provided for you to simply understand the utterance(s) made before the new utterance so as to help you better annotate the new utterance.
Thus, please do not annotate the entire conversation but annotate only the new utterance by determining its appropriate speech act label (s). Please provide the final answer directly with no reasoning steps.
If the new utterance fits multiple categories, list all that apply. Ensure that your final answer clearly identifies whether each of the of the speech act labels (0-11) are applicable (1) or not applicable (0) to the new utterance.
For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of each label as follows:

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
"""


def calculate_speech_acts_labels(utts, topic, openAIKEY, model_type, model, ctx):

    prompt = ini + speech_act_labels + final
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


def calculate_speech_acts(
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
    """Annotates the utterances in a discussion using speech act labels based on the frameworks proposed by Fournier-Tombs and MacKenzie (2021) and Zhang, Culbertson and Paritosh (2017).
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
        dict: A dictionary mapping discussion IDs to lists of per-utterance speech act feature dictionaries.
              Each entry is keyed by utterance ID (e.g., "utt_0") and contains extracted features.

    """

    validateInputParams(model_type, openAIKEY, model_path)
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None
    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    #
    speechactslabels_llm_output_dict = {}
    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        print("Speech acts labels-Proccessing discussion: ", disc_id, " with LLM")
        #
        speechacts_features = calculate_speech_acts_labels(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        speechactslabels_llm_output_dict[disc_id] = speechacts_features
        #
        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        speechactslabels_llm_output_dict,
        "llm_output_speechact_per_response",
        disc_id,
        timestr,
    )

    """
    with open("llm_output_speechact_per_response_.json", encoding="utf-8") as f:
        spcact_labels = json.load(f)
    speechactslabels_llm_output_dict=spcact_labels
    """

    speechact_per_response = {}
    for disc_id, turnAnnotations in speechactslabels_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                print("LLM output with missing speech act label, skipping response\n")
                print(label)
                counter += 1
                continue
            
            feature = {}
            
            feature=extractFeature(feature , label)
            if feature == -1:
                counter+=1
                continue

            key_iter = "utt_" + str(counter)
            ut_dict[key_iter] = [feature]
            counter += 1
            
        if disc_id in speechact_per_response:
            speechact_per_response[disc_id].append(ut_dict)
        else:
            speechact_per_response[disc_id] = [ut_dict]

    save_dict_2_json(
        speechact_per_response, "speechact_per_utterance", disc_id, timestr
    )
    return speechact_per_response

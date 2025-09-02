import time

from DiscQuA.utils import (
    dprint,
    extractFeature,
    getModel,
    getUtterances,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

from .DisputeTacticsCl import DisputeTactics


def calculate_dispute_tactics(
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
    """Annotates the utterances in a discussion using the dispute tactics labels based on the frameworks proposed by De Kock and Vlachos (2022, December).

     Args:
         message_list (list[str]):  The list of utterances in the discussion.
         speakers_list (list[str]): The corresponding list of speakers for each utterance.
         disc_id (str): Unique identifier for the discussion.
         openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
         model_type (str): Language model type to use, either "openai" or "llama" or "transformers". Defaults to "openai".
         model_path (str): Path to the model, used only for model_type "llama" or "transformers". Defaults to "".
         gpu (bool): A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.
         ctx (int): Number of previous utterances to include as context for each input. Defaults to 1.
         device(str): The device to load the model on. If None, the device will be inferred. Defaults to auto.


    Returns:
        dict[str, list[dict[str, dict[str, int]]]]: A mapping from the discussion ID to a list
        of dictionaries, each representing dispute tactic features per utterance (e.g., "utt_0").




    """
    validateInputParams(model_type, openAIKEY, model_path)
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    dispute_tactics_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = message_list[0]
        dprint("info", f"Discpute tactics-Proccessing disc: {disc_id} with LLM")

        dispute = DisputeTactics(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx
        )
        dispute_features = dispute.calculate_dispute_tectics()
        dispute_tactics_llm_output_dict[disc_id] = dispute_features

        sleep(model_type)
    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        dispute_tactics_llm_output_dict, "llm_dispute_tactics", disc_id, timestr
    )

    """        
    with open("llm_dispute_tactics_.json", encoding="utf-8") as f:
        d = json.load(f)
    dispute_tactics_llm_output_dict=d
    """

    dispute_tactics_per_utt = {}
    for disc_id, turnAnnotations in dispute_tactics_llm_output_dict.items():
        counter = 0
        ut_dict = {}
        for label in turnAnnotations:
            if label == -1:
                dprint(
                    "info",
                    "LLM output for utterance is ill-formatted, skipping utterance\n",
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

        if disc_id in dispute_tactics_per_utt:
            dispute_tactics_per_utt[disc_id].append(ut_dict)
        else:
            dispute_tactics_per_utt[disc_id] = [ut_dict]

    save_dict_2_json(
        dispute_tactics_per_utt,
        "dispute_tactics_features_per_utterance",
        disc_id,
        timestr,
    )
    return dispute_tactics_per_utt

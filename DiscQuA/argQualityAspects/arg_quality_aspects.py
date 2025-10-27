import sys
import time

from discqua.utils import (
    dprint,
    extractFeature,
    getModel,
    getUtterances,
    isValidResponse,
    save_dict_2_json,
    sleep,
    validateInputParams,
)

from .arg_qual_dimensions import AQualityDimensions


def arg_dimensions(
    message_list,
    speakers_list,
    msgsid_list,
    disc_id,
    conver_topic,
    openAIKEY,
    model_type="openai",
    model_path="",
    gpu=False,
    ctx=1,
    dimension="logic",
    device="auto",
):
    """Evaluates the quality of argumentation in each utterance of a discussion according to the taxonomy proposed by Wachsmuth et al. (2017).
       Each utterance is assessed across multiple argument quality dimensions, each scored on a scale from 1 (low) to 3 (high).

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        msgsid_list (list[str]) : List of messages ids corresponding to each utterance.
        disc_id (str): Unique identifier for the discussion.
        conver_topic(str): The topic of conversation.
        openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
        model_type (str): Language model type to use, either "openai" or "transformers". Defaults to "openai".
        model_path (str): Path to the model, used only for model_type "transformers". Defaults to "".
        gpu (bool): A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.
        ctx (int): Number of previous utterances to include as context for each input. Defaults to 1.
        dimension (str): logic, rhetoric, dialectic, overall.
        device(str): The device to load the model on. If None, the device will be inferred. Defaults to auto.

    Returns:
        dict: A dictionary mapping the discussion ID to a list of per-utterance argument quality annotations.
              Each utterance is represented by message ID key and contains the evaluated quality
              dimensions as subfields.
    """

    validateInputParams(model_type, openAIKEY, model_path, message_list, msgsid_list)
    if dimension not in ["logic", "rhetoric", "dialectic", "overall"]:
        print(
            f"Invalid argument quality dimension: {dimension}. Expected one of ",
            "logic",
            "rhetoric",
            "dialectic",
            "overall",
        )
        sys.exit(1)

    dprint("info", f"Building corpus of: {len(message_list)} utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    argqualitydimensions_scores_llm_output_dict = {}

    try:
        utterances, speakers = getUtterances(
            message_list, speakers_list, disc_id, replyto_list=[]
        )
        conv_topic = conver_topic
        dprint(
            "info", f"Argument Quality Dimensions-Proccessing disc: {disc_id} with LLM"
        )

        argqualdimensions = AQualityDimensions(
            utterances, conv_topic, openAIKEY, model_type, llm, ctx, dimension
        )

        argument_quality_dimensions_features = (
            argqualdimensions.argquality_dimensions_scores()
        )

        argqualitydimensions_scores_llm_output_dict[disc_id] = (
            argument_quality_dimensions_features
        )

        sleep(model_type)

    except Exception as e:
        print("Error: ", e)
        print(disc_id)

    save_dict_2_json(
        argqualitydimensions_scores_llm_output_dict,
        "llm_output_md_aq",
        disc_id,
        timestr,
    )

    """        
    with open("llm_output_md_aq_", encoding="utf-8") as f:
        md_aq = json.load(f)
    argqualitydimensions_scores_llm_output_dict=md_aq
    """
    arq_dim_per_disc = {}
    for disc_id, turnAnnotations in argqualitydimensions_scores_llm_output_dict.items():
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

            if dimension == "overall":
                parts = label.split("overall argument quality is:")
                value = isValidResponse(parts)
                if value == -1:
                    dprint(
                        "info",
                        "LLM output for utterance is ill-formatted, skipping utterance\n",
                    )
                    dprint("info", label)
                    counter += 1
                    continue
                feature["overall"] = value

            else:
                feature = extractFeature(feature, label)
                if feature == -1:
                    counter += 1
                    continue
            # key_iter = "utt_" + str(counter)
            key_iter = msgsid_list[counter]
            ut_dict[key_iter] = [feature]
            counter += 1
        if disc_id in arq_dim_per_disc:
            arq_dim_per_disc[disc_id].append(ut_dict)
        else:
            arq_dim_per_disc[disc_id] = [ut_dict]

    save_dict_2_json(
        arq_dim_per_disc, dimension + "_md_argqual_per_utt", disc_id, timestr
    )
    return arq_dim_per_disc

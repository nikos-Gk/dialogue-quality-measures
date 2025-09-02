import time

from DiscQuA.utils import (
    dprint,
    getModel,
    getUtterances,
    isValidResponse,
    save_dict_2_json,
    validateInputParams,
)

from .argument_quality_overall import OAQuality


def calculate_overall_arg_quality(
    message_list,
    speakers_list,
    disc_id,
    openAIKEY,
    model_type="openai",
    model_path="",
    mode="real",
    gpu=False,
    device="auto",
):
    """Calculates the overall argument quality score for a given discussion using a specified language model.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        disc_id (str): Unique identifier for the discussion.
        openAIKEY (str): OpenAI API key, required if using OpenAI-based models.
        model_type (str): Language model type to use, either "openai" or "llama" or "transformers". Defaults to "openai".
        model_path (str): Path to the model, used only for model_type "llama" or "transformers". Defaults to "".
        mode (str): "rating" for a rating label or "real" for a real score. Defaults to "real".
        gpu (bool): A boolean flag; if True, utilizes GPU (when available); otherwise defaults to CPU. Defaults to False.
        device(str): The device to load the model on. If None, the device will be inferred. Defaults to auto.

    Returns:
        dict: A dictionary mapping the discussion ID to its overall argument quality score.
    """

    validateInputParams(model_type, openAIKEY, model_path)
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    llm = None

    if model_type == "llama" or model_type == "transformers":
        llm = getModel(model_path, gpu, model_type, device)

    ovargquality_scores_llm_output_dict = {}
    #
    utterances, speakers = getUtterances(
        message_list, speakers_list, disc_id, replyto_list=[]
    )
    conv_topic = message_list[0]
    #
    dprint(
        "info", f"Overall Argument Quality-Proccessing discussion: {disc_id} with LLM "
    )
    ovargqual = OAQuality(utterances, conv_topic, openAIKEY, mode, model_type, llm)
    ovargument_quality_scores_features = ovargqual.calculate_ovargquality_scores()
    ovargquality_scores_llm_output_dict[disc_id] = ovargument_quality_scores_features
    #
    save_dict_2_json(
        ovargquality_scores_llm_output_dict, "llm_output_ovrall_aq", disc_id, timestr
    )
    """
    with open(
        "llm_output_ovrall_aq_",
        encoding="utf-8",
    ) as f:
        oaq = json.load(f)
    ovargquality_scores_llm_output_dict = oaq
    """

    oaq_dim_per_disc = {}
    for disc_id, turnAnnotations in ovargquality_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1:
                dprint(
                    "info",
                    "LLM output with missing overall argument quality , skipping discussion\n",
                )
                dprint("info", label)
                continue
            parts = label.split(
                "average overall quality of the arguments presented in the above discussion is:"
            )
            value = isValidResponse(parts)
            if value == -1:
                dprint(
                    "info",
                    "LLM output with missing overall argument quality , skipping discussion\n",
                )
                dprint("info", label)
                continue

            oaq_dim_per_disc[disc_id] = value

    save_dict_2_json(oaq_dim_per_disc, "ovrall_argqual_per_disc", disc_id, timestr)
    return oaq_dim_per_disc

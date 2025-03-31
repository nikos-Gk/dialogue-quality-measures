import json
import os
import sys

import pandas as pd
from tqdm import tqdm

MODERATOR_NAME = "moderator"


def filter_df_by_key(key, df_to_filter, column):
    mask = key == df_to_filter[column]
    matched_df = df_to_filter[mask]
    return matched_df


def extract_info(column, df_to_extract, warning_message):
    unq = list(df_to_extract[column].unique())
    if len(unq) != 1:
        print(unq)
        print(warning_message)
    return unq[0]


def extract_conv_info(matched_df, unique_user_ids, conv_id):
    time_stamp = matched_df["timestamp_conv"].unique()[0]
    ctx_length_conv = matched_df["ctx_length_conv"].unique()[0]
    conv_variant = matched_df["conv_variant"].unique()[0]
    has_moderator = matched_df["is_moderator"].any()
    llm_model = sorted(list(matched_df["model"].unique()))
    if has_moderator and MODERATOR_NAME not in unique_user_ids:
        print(
            "WARNING: has_moderator true but no moderator in user ids, conv_id: ",
            conv_id,
        )
    return time_stamp, ctx_length_conv, conv_variant, llm_model


def get_user_prompts(unique_user_ids, df_to_extract_from, conv_id):
    usersprompt_dict = {}
    for user in unique_user_ids:
        matcheduser_df = filter_df_by_key(user, df_to_extract_from, "user")

        userprompt_unique = (
            extract_info(
                "user_prompt",
                matcheduser_df,
                f"WARNING: different prompts for the same user, user_id,conv_id :{(user,conv_id)}",
            ),
        )
        usersprompt_dict[user] = userprompt_unique
    return usersprompt_dict


def get_conv_messages(df_to_extract_from, conv_id):
    message_order = sorted(list(df_to_extract_from["message_order"].unique()))
    message_dict = {}
    for order in message_order:
        matchedmessageorder_df = filter_df_by_key(
            order, df_to_extract_from, "message_order"
        )

        turnuser_unq = extract_info(
            "user",
            matchedmessageorder_df,
            f"WARNING: different user for the same message: conv_id, turn_id: {(conv_id,order)}",
        )
        message_unq = extract_info(
            "message",
            matchedmessageorder_df,
            f"WARNING: different messages for the same user, conv_id, turn_number: {(turnuser_unq,conv_id,order)}",
        )
        model_unq = extract_info(
            "model",
            matchedmessageorder_df,
            f"WARNING: different model for the same user and turn_id: {(conv_id, order)}",
        )
        message_id_unq = extract_info(
            "message_id",
            matchedmessageorder_df,
            f"WARNING: different message_id for the same user and turn_id: conv_id, turn_id {(conv_id,order)}",
        )
        message_dict[str(order)] = (
            turnuser_unq,
            message_unq,
            model_unq,
            str(message_id_unq),
        )
    return message_dict


def convert_csv_to_json(input_file):
    if not os.path.exists(input_file):
        print(input_file)
        print("input file does not exist. Exiting")
        sys.exit(1)

    dataset_df = pd.read_csv(input_file)

    unique_conv_ids = sorted(list(dataset_df["conv_id"].unique()))

    for conv_id in tqdm(unique_conv_ids):
        matched_df = filter_df_by_key(conv_id, dataset_df, "conv_id")

        unique_user_ids = sorted(list(matched_df["user"].unique()))

        time_stamp, ctx_length_conv, conv_variant, llm_model = extract_conv_info(
            matched_df, unique_user_ids, conv_id
        )

        usersprompt_dict = get_user_prompts(unique_user_ids, matched_df, conv_id)
        message_dict = get_conv_messages(matched_df, conv_id)

        sorted_keys = sorted([int(key) for key in message_dict.keys()])
        message_list = [message_dict[str(key)] for key in sorted_keys]

        data = {
            "id": conv_id,
            "timestamp": time_stamp,
            "users": unique_user_ids,
            "moderator": True if MODERATOR_NAME in unique_user_ids else False,
            "ctx_length": int(ctx_length_conv),
            "conv_variant": conv_variant,
            "llm_model": llm_model,
            "user_prompts": usersprompt_dict,
            "logs": message_list,
        }

        cwd = os.getcwd()
        file_path_with_mod = os.path.join(cwd, "with_mod")
        file_path_no_mod = os.path.join(cwd, "no_mod")

        file_path = None
        if not os.path.exists(file_path_with_mod) and MODERATOR_NAME in unique_user_ids:
            os.makedirs(file_path_with_mod)
            file_path = os.path.join(file_path_with_mod, conv_id + ".json")
        elif os.path.exists(file_path_with_mod) and MODERATOR_NAME in unique_user_ids:
            file_path = os.path.join(file_path_with_mod, conv_id + ".json")

        if (
            not os.path.exists(file_path_no_mod)
            and MODERATOR_NAME not in unique_user_ids
        ):
            os.makedirs(file_path_no_mod)
            file_path = os.path.join(file_path_no_mod, conv_id + ".json")
        elif os.path.exists(file_path_no_mod) and MODERATOR_NAME not in unique_user_ids:
            file_path = os.path.join(file_path_no_mod, conv_id + ".json")

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

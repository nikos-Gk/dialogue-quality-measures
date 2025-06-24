import json
import math
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
    if warning_message != "-" and len(unq) != 1:
        print(unq)
        print(warning_message)
    return unq[0]


def extract_conv_info(matched_df, unique_user_ids, conv_id):
    time_stamp = matched_df["timestamp_conv"].unique()[0]
    has_moderator = matched_df["is_moderator"].any()
    llm_model = sorted(list(matched_df["model"].unique()))
    if has_moderator and MODERATOR_NAME not in unique_user_ids:
        print(
            "WARNING: has_moderator true but no moderator in user ids, conv_id: ",
            conv_id,
        )
    return time_stamp, llm_model


def get_conv_messages(df_to_extract_from, conv_id, replyto_bool):
    message_order = sorted(list(df_to_extract_from["message_order"].unique()))
    message_dict = {}
    for order in message_order:
        matchedmessageorder_df = filter_df_by_key(
            order, df_to_extract_from, "message_order"
        )

        turnuser_unq = extract_info(
            "user",
            matchedmessageorder_df,
            f"WARNING: different user for the same message: conv_id, turn_id: {(conv_id, order)}",
        )
        message_unq = extract_info(
            "message",
            matchedmessageorder_df,
            f"WARNING: different messages for the same user, conv_id, turn_number: {(turnuser_unq, conv_id, order)}",
        )
        model_unq = extract_info(
            "model",
            matchedmessageorder_df,
            "-",
        )
        message_id_unq = extract_info(
            "message_id",
            matchedmessageorder_df,
            f"WARNING: different message_id for the same user and turn_id: conv_id, turn_id {(conv_id, order)}",
        )

        replyto_unq = "-"

        if replyto_bool:
            replyto_unq = extract_info(
                "reply_to",
                matchedmessageorder_df,
                f"WARNING: different reply_to for the same user and turn_id: conv_id, turn_id {(conv_id, order)}",
            )

            if math.isnan(replyto_unq):
                replyto_unq = "-"

        message_dict[str(order)] = (
            turnuser_unq,
            message_unq,
            model_unq,
            str(message_id_unq),
            str(replyto_unq),
        )
    return message_dict


def convert_csv_to_json(input_file):
    if not os.path.exists(input_file):
        print(input_file)
        print("input file does not exist. Exiting")
        sys.exit(1)

    dataset_df = pd.read_csv(input_file)
    replyto_bool = False

    columns2have = [
        "conv_id",
        "timestamp_conv",
        "user",
        "message",
        "model",
        "is_moderator",
        "message_id",
        "message_order",
        "reply_to",
    ]

    for clm in columns2have:
        if clm not in dataset_df.columns and clm != "reply_to":
            print(f"missing column in csv file: {clm}")
            sys.exit(1)

    if "reply_to" in dataset_df.columns:
        replyto_bool = True

    unique_conv_ids = sorted(list(dataset_df["conv_id"].unique()))

    for conv_id in tqdm(unique_conv_ids):
        matched_df = filter_df_by_key(conv_id, dataset_df, "conv_id")

        unique_user_ids = sorted(list(matched_df["user"].unique()))

        time_stamp, llm_model = extract_conv_info(matched_df, unique_user_ids, conv_id)

        message_dict = get_conv_messages(matched_df, conv_id, replyto_bool)

        sorted_keys = sorted([int(key) for key in message_dict.keys()])
        message_list = [message_dict[str(key)] for key in sorted_keys]

        data = {
            "id": str(conv_id),
            "timestamp": time_stamp,
            "users": unique_user_ids,
            "moderator": True if MODERATOR_NAME in unique_user_ids else False,
            "llm_model": llm_model,
            "logs": message_list,
        }

        cwd = os.getcwd()
        file_path_with_mod = os.path.join(cwd, "with_mod")
        file_path_no_mod = os.path.join(cwd, "no_mod")

        file_path = None
        conv_id = str(conv_id)
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

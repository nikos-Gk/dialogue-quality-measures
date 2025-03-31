# -*- coding: utf-8 -*-
import time

import openai

ini = """Below are two sets of linguistic features for disagreement levels and non-disagreement labels."""

disagreement_levels = """\n\n
Disagreement Levels:

Level 0: Name calling/hostility - Direct insults, or use of an equally hostile tone or language.

Level 1: Ad hominem/ad argument - An attack to the person, often used to attempt to discredit an opponent (e.g., "I know better than you; you do not have a physics degree"). This category was extended to include "ad argument", where someone insults another's argument (e.g., "that is ridiculous") without responding to its content.

Level 2: Attempted derailing/off-topic - This category was added to address comments which are unrelated to the current line of discussion and fail to further the argument, while still being argumentative. This category is also represented in the Talk page taxonomy of Viegas et al. (2007). It was assigned a lower level as it can be detrimental to the argument, by taking focus away from the main topic.

Level 3: Policing the discussion - Graham (2008) referred to this as "responding to tone" with the description "responses to the writing, rather than the writer. The lowest form of these is to disagree with the author’s tone". By expanding the definition to "policing the discussion", the idea is to include people saying "You’ve said that before", telling people to "calm down", correcting spelling errors, or citing discussion policy (i.e. "no personal attacks"). It ignores the argument’s content.

Level 4a: Stating your stance - Graham (2008) referred to this as "contradiction", with the description "to state the opposing case, with little or no supporting evidence".

Level 4b: Repeated argument - Level added to describe re-stating an argument used before, potentially using different words, but without furthering the discussion.

Level 5: Counterargument - Described as "contradiction plus new reasoning and/or evidence", which does not directly address the opponent’s argument.

Level 6: Refutation - Directly responding to the argument and explaining why it is mistaken, using new evidence or reasoning.

Level 7: Refuting the central point - Graham (2008) notes that "Truly refuting something requires one to refute its central point, or at least one of them." Unfortunately, the central point can be quite subjective and difficult to recognize for non-experts, and may change throughout a conversation. As such, we use this category as a prime example of a “good” refutation, which is of course still subjective and may be rolled up into Level 6.
"""


non_disagreement_labels = """\n\n
Non-Disagreement Labels:

Label A: Bailing out - An indication that a person is giving up on a discussion and will no longer engage.

Label B: Contextualisation - In the first utterance, a person "sets the stage" by describing which aspect of the discussion they are challenging. This does not directly disagree with anyone, and is therefore a non-disagreement move.

Label C: Asking questions - Seeking to understand another person’s opinion better. This does not include rhetorical questions, which are generally disagreement moves.

Label D: Providing clarification - Answering questions or providing information which seeks to create understanding, rather than only furthering a point.

Label E: Suggesting a compromise - An attempt to find a midway between one’s own point and the opposer’s.

Label F: Coordinating - For example, Wikipedia Talk pages are primarily used to for goal-oriented discussions, to coordinate edits to a page. As part of disagreement threads, there is often also some discussion of these edits. This can signal that a compromise has been found.

Label G: Conceding / recanting - An explicit admission that an interlocutor is willing to relinquish their point.

Label H: I don't know - Admitting that one is uncertain. This signals that an person is receptive to the idea that there are unknowns which may impact their argument.

Label I: Other - For utterances not covered by any other class, for instance, social niceties.
"""

final = """\n\n
Given a conversation history (which can be empty if the new utterance is the first utterance made in the conversation), please analyze a new utterance from a user (identified by a unique user_id) in a conversation discussing with others about this potentially controversial post.
Post: {post} 
Each utterance is a brief Reddit-based comment responding to the post and other users' comments on it.
    
*CONVERSATION HISTORY*: "{conv_history}"

*NEW UTTERANCE*: "{utterance}"

Noteworthy, the conversation history is provided for you to simply understand the utterances made before the new utterance so as to help you better annotate the new utterance.

Thus, please do not annotate the entire conversation but annotate only the new utterance by determining its appropriate disagreement level and/or non-disagreement label(s). Please provide the final answer directly with no reasoning steps.

If the new utterance fits multiple categories, list all that apply. Ensure that your final answer clearly identifies whether each of the disagreement levels (0-7) and non-disagreement labels (A-I) are applicable (1) or not applicable (0) to the new utterance.

For clarity, your response should succinctly be presented in a structured list format, indicating the presence (1) or absence (0) of each level and label as follows:

- Level 0: [1/0]
- Level 1: [1/0]
- Level 2: [1/0]
- Level 3: [1/0]
- Level 4a: [1/0]
- Level 4b: [1/0]
- Level 5: [1/0]
- Level 6: [1/0]
- Level 7: [1/0]
- Label A: [1/0]
- Label B: [1/0]
- Label C: [1/0]
- Label D: [1/0]
- Label E: [1/0]
- Label F: [1/0]
- Label G: [1/0]
- Label H: [1/0]
- Label I: [1/0]
"""


class DisputeTactics:
    def __init__(self, utterances, conv_topic, openaiKey):
        self.utterances = utterances
        self.conv_topic = conv_topic
        self.openaiKey = openaiKey

    def prompt_gpt4(self, prompt):
        openai.api_key = self.openaiKey

        ok = False
        counter = 0
        while (
            not ok
        ):  # to avoid "ServiceUnavailableError: The server is overloaded or not ready yet."
            counter = counter + 1
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0,
                )
                ok = True
            except Exception as ex:
                print("error", ex)
                print("sleep for 5 seconds")
                time.sleep(5)
                if counter > 10:
                    return -1
        return response["choices"][0]["message"]["content"]

    def calculate_dispute_tectics(self):
        prompt = ini + disagreement_levels + non_disagreement_labels + final
        annotations_ci = []
        conv_hist = ""
        for index, utt in enumerate(self.utterances):
            text = utt.text
            speaker = utt.get_speaker().id
            if index == 0:
                conv_hist = ""
            else:
                conv_hist = (
                    conv_hist
                    + "\n"
                    + "<user_name="
                    + self.utterances[index - 1].get_speaker().id
                    + "\n"
                    + self.utterances[index - 1].text
                    + "\n"
                )
            try:
                formatted_prompt = prompt.format(
                    utterance="<user_name=" + speaker + ">" + "\n" + text,
                    conv_history=conv_hist,
                    post=self.conv_topic,
                )
                response_text = self.prompt_gpt4(formatted_prompt)
                annotations_ci.append(response_text)
            #                print(formatted_prompt)
            except Exception as e:  # happens for one instance
                print("Error: ", e)
                annotations_ci.append(-1)
        return annotations_ci

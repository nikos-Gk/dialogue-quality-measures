# Measures

Discussion Quality Aspects

# Installation
```python
git clone https://github.com/nikos-Gk/dialogue-quality-measures.git
cd dialogue-quality-measures
git fetch --all
git checkout package
conda create --name discMeasuresEnv python=3.12
conda activate discMeasuresEnv
pip install -e .
python -m spacy download en_core_web_sm
```

# Usage

In the main.py file you can find examples of usage.

# Examples

## convert a csv to json files for further processing
## Convert a csv with discussions to a JSON format. 
## Output:
### disc_directory: directory with json files
## Required columns: 
### conv_id: conversation id, 
### timestamp_conv: timestamp of conversation, 
### user: speaker name, 
### message: message text, 
### model: (LLM) model that generated the discussion (if applicable; otherwise -), 
### is_moderator: boolean indicating message sent by a moderator, 
### message_id, 
### message_order=the order of the message within the discussion, 
### reply_to: message_order that the utterance is replying to (if applicable; otherwise -).  


```python
from DiscQuA import convert_csv_to_json
convert_csv_to_json(
    input_file=csv_path
)
```

disc_directory: directory with json files


## structure features
## structure features (discussion and turn level): 140 features derived from the structure of the online discussions (drawn from Zhang et al., 2018). 
### Example: %_NONZERO[OUTDEGREE over C->c REPLIES] is the proportion of participants that have replied. 
## Reciprocity motifs: proxies for the engagement (interactional patterns) in the discussion. 
#### Reciprocity motif: when the target of a reply returns to respond to the replier.
#### External reciprocity motif: captures the tedency of a comment to draw responses from actors beyond its explicit target.
#### Dyadic interaction motif: dyadic interactions between pairs of commenters across the entire descussion.
#### Incoming triads: pairs of responses received by a commenter from two other actors.
#### Outgoing triads: when a speaker responds to two other discussion participants.


```python
from DiscQuA import calculate_structure_features
 calculate_structure_features(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## turn taking (turn level)
### calculate_balanced_participation: 
#### entropy of number of messages and words per message. A value close to 1 indicates that all participants contribute equally to the discussion, while values approaching 0 suggest that one participant dominates the discussion.

### make_visualization: 
#### a PNG file that visually represents how discussants alternate in speaking during the discussion.

```python
from DiscQuA import calculate_balanced_participation, make_visualization

    calculate_balanced_participation(
        input_directory=disc_directory,
        moderator_flag=False,
    )

    make_visualization(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## language features
## language features (discussion and turn level)
### Language features based on common words (comwords), stopwords and content words (contwords) between the messages in a discussion. 

## Labels: Let utt_i, and utt_i+1 two consecutive utterances in the discussion. 
### n_comwords, n_stopwords, n_contwords: |utt_i ∩ utt_i+1|.
### reply_fra_comwords, reply_fra_stopwords, reply_fra_contwords: |utt_i ∩ utt_i+1|/|utt_i+1|
### op_fra_comwords, op_fra_stopwords,op_fra_contwords:  |utt_i ∩ utt_i+1|/|utt_i|
### jac_comwords, jac_stopwords, jac_contwords: |utt_i ∩ utt_i+1|/|utt_i ∪ utt_i+1|


```python
from DiscQuA import calculate_language_features

     calculate_language_features(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## controversy
## controversy: a quantification of the level of controversy in a discussion by examining the sample standard of the distribution of sentiment scores attributed to the discussion comments by a pretrained BERT model.
### Labels
#### bert_sent_per_comment_unnorm: the weigted mean sentiment scores attributed to each discussion comment by the BERT model.
#### bert_sent_per_comment_norm: the normalized weigted mean sentiment scores attributed to each discussion comment by the BERT model.
#### controversy_per_disc_unorm: sample standard deviation of the sentiment scores assigned by BERT to the discussion comments.
#### controversy_per_disc_norm: sample standard deviation of the normalized sentiment scores assigned by BERT to the discussion comments.


```python
from DiscQuA import calculate_controversy

    calculate_controversy(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## coherence

### coherence conversation with openai

```python
from DiscQuA import calculate_coherence_conversation

      calculate_coherence_conversation(
        input_directory=disc_directory,
        openAIKEY="your-openai-key",
        model_type="openai",
        model_path="",
        gpu=False,
        moderator_flag=False,
        )
```

### coherence conversation with llama, local inference. Using llama-cpp-python that supports gguf models

```python
def test_coherence_conversation():
    calculate_coherence_conversation(
        input_directory=disc_directory,
        openAIKEY="",
        model_type="llama",
        model_path="/path/to/model/llama-2-13b-chat.Q5_K_M.gguf",
        gpu=True,
        moderator_flag=False,
    )
```


### coherence response 

```python
from DiscQuA import calculate_coherence_response

       calculate_coherence_response(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

### coherence ecoh 

```python
from DiscQuA import calculate_coherence_ecoh

       calculate_coherence_ecoh(
        input_directory=disc_directory,
        moderator_flag=False,
        device="cpu",
    )
```

## diversity

```python
from DiscQuA import calculate_diversity

      calculate_diversity(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## empathy

```python
from DiscQuA import dialogicity

      dialogicity(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## engagement

```python
from DiscQuA import engagement

      engagement(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## informativeness

```python
from DiscQuA import informativeness

      informativeness(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## persuasiveness

```python
from DiscQuA import persuasiveness

      persuasiveness(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## toxicity

```python
from DiscQuA import toxicity

      toxicity(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```


## power or status and social bias

### power or status: For each participant in a discussion, analyzes the extent to which they mirror the linguistic style of those they are responding to, as well as how much others, replying to them, imitate their linguistic style.
#### Linguistic Style: articles, auxiliary verbs, conjunctions, adverbs, personal pronouns, impersonal pronouns, prepositions, quantifiers.
#### Aggregate measures
##### agg1: aggegate measure that does not use smoothing assumptions.
##### agg2: aggegate measure that uses smoothing assumptions.
##### agg3: aggegate measure that uses smoothing assumptions.


```python
from DiscQuA import calculate_coordination_per_discussion

       calculate_coordination_per_discussion(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

### social bias
### social bias: Pragmatic frames in which people project social biases and stereotypes onto others.
### Labels:
##### Label 0-Offensiveness: Comment is considered offensive, disrespectful, or toxic to anyone/someone.
##### Label 1-Intent to offend: The intent of the comment is to be offensive/disrespectful to anyone.
##### Label 2-Lewd: The comment contain or allude to sexual content/acts.
##### Label 3-Group implications:The comment imply offense/disrespect to an identity-related group of people (e.g., muslims).
##### Label 4-In group language: The comment imply offense/disrespect to an identity-related group of people and not just an insult to an individual or non-identity-related group of people and the author of the comment sound like they belong to the same social demographic group that is targeted.

```python
from DiscQuA import calculate_social_bias

    calculate_social_bias(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## politeness
## Politeness strategies from Danescu-Niculescu-Mizil et al. (2013, August).
### Strategies (discussion and turn level):
#### Please: The presence of the word ‘please’ in the sentence (e.g., Could you please say more. . .).
#### Start with ‘Please’: The sentence starts with the word ‘please’ (e.g., Please do not remove warnings . . .).
#### Has subject hedge: Any subject in the sentence depends on a hedge word (e.g., I suggest we start with . . .).
#### Use of ‘by the way’: The phrase ‘by the way’ is used in the sentence (e.g., By the way, where did you find . . .).
#### Hedge words: Any word in the sentence is a hedge word.
#### Assert factuality: Words that assert factuality, like ‘in fact,’ ‘actually,’ or ‘really’ (e.g., In fact you did link).
#### Start with deference: The sentence starts with deferential words like ‘great,’ ‘good,’ or ‘nice’ (e.g., Nice work so far on your rewrite).
#### Gratitude: Expressions of gratitude, like ‘thank’ or ‘thanks’ (e.g., I really appreciate that you’ve done them).
#### Apologising: Apologetic expressions like ‘sorry,’ or ‘I apologize’ (e.g.,  Sorry to bother you . . .).
#### 1st person plural: First-person plural pronouns (e.g.,Could we find a less complex name . . .).
#### 1st person pronouns: First-person singular pronouns (e.g., It is my view that ...).
#### Start with 1st person: The sentence starts with a first-person singular pronoun (e.g.,  I have just put the article . . .).
#### 2nd person pronouns: Second-person pronouns like (e.g.,  But what’s the good source you have in mind?).
#### Start with 2nd person:  The sentence starts with a second-person pronoun (e.g., You’ve reverted yourself . . .).
#### Start with greeting: The sentence starts with a greeting word (e.g., Hey, I just tried to . . .).
#### Starts with question: The sentence starts with a question word like ‘what,’ ‘why,’ ‘who,’ or ‘how’ (What is your native language?).
#### Starts with conjunction: The sentence starts with a conjunction or transition word like ‘so,’ ‘then,’ ‘and,’ ‘but,’ or ‘or’ (e.g., So can you retrieve it or not?).
#### Positive sentiment words: The presence of positive sentiment words (e.g.,  Wow! / This is a great way to deal. . .).
#### Negative sentiment words: The presence of negative sentiment words (e.g., If you’re going to accuse me . . .).
#### Subjunctive words: The use of subjunctive mood words like ‘could’ or ‘would’ when preceded by ‘you’ (e.g., Could/Would you . . .).
#### Indicative words: The use of indicative mood words like ‘can’ or ‘will’ when preceded by ‘you’ (e.g., Can/Will you . . .).




```python
from DiscQuA import calculate_politeness

    calculate_politeness(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## collaboration

```python
from DiscQuA import calculate_collaboration

    calculate_collaboration(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## Argument Quality Aspects
## Argument Quality Aspects (turn level): 
### Assess the quality of the employed argumentation in line with the Wachsmuth et al. (2017) taxonomy.
#### Label 1a-Local Acceptability: Premises of the argument worthy of being believed.
#### Label 1b-Local Relevance: Premises support/attack the conclusion of the argument. 
#### Label 1c-Local Sufficiency: Premises enough to draw the conclusion of the argument. 
#### Label 1_overall-Cogency: Argument has locally acceptable, relevant, and sufficient premises. 
#### Label 2a-Credibility: Argument makes the author worthy of credence. 
#### Label 2b-Emotional Appeal: Argument creates emotions tha make the audience open to it. 
#### Label 2c-Clarity: Argument avoids deviation from the issue, and uses correct and unambiguous language. 
#### Label 2d-Appropriateness: Language proportional to the issue, supports credibility and emotions. 
#### Label 2e-Arrangement: Discussion topic, arguments and conlusions are presented in the right order. 
#### Label 2_overall-Effectiveness: Argument persuades the audience. 
#### Label 3a-Global Acceptability: Audience accepts the use of the argument. 
#### Label 3b-Global Relevance: Argument helps arrive at an agreement. 
#### Label 3c-Global Sufficiency: Argument adequately rebuts the anticipated counter-arguments to its conclusion. 
#### Label 3_overall-Reasonableness: Argument is (globally) acceptable, relevant, and sufficient.
#### Label 4-overall quality: Argumentation quality in total.



```python
from DiscQuA import calculate_arg_dim

    calculate_arg_dim(
        input_directory=disc_directory,
        openAIKEY="", 
        moderator_flag=False
    )

```

## Overall Argument Quality
## Overall Argument Quality (discussion level): 
### Assess the average overall quality of the employed argumentation. 
### Mode="real" ("rating") assigns a real (integer) number on a scale from 1 to 5 (1 to 3).




```python
from DiscQuA import calculate_overall_arg_quality

    calculate_overall_arg_quality(
        input_directory=disc_directory,
        openAIKEY="",
        mode="real",
        moderator_flag=False,
    )


```
## Dispute Tactics
## Dispute Tactics (turn-level): linguistic features for disagreement levels and coordination labels.
### Disagreement levels:
#### Level 0-Name calling/hostility: Comment with direct insults, or use of an equally hostile tone or language. 
#### Level 1-Ad hominem/ad argument: Comment that attacks other users to discredit them or their arguments without addressing the content. 
#### Level 2-Attempted derailing/off-topic: Comment that is unrelated to the current line of discussion.
#### Level 3-Policing the discussion: Comment that does not address (opposing) arguments’ content but mainly try to "policy" the discussion (e.g., telling people to "calm down").
#### Level 4a-Stating your stance: Comment that states an opposing view (stance), with little or no supporting evidence.
#### Level 4b-Repeated argument: Comment that re-states a previously expressed argument, potentially using different words, but without furthering the discussion.
#### Level 5-Counterargument: Comment that states an opposing view (stance), while providing supporting reasoning and/or evidence.
#### Level 6-Refutation: Comment that responds to an argument and explains why it is mistaken, using new evidence or reasoning.
#### Level 7-Refuting the central point: Comment that directly refutes the central point of an argument, explaining why it is mistaken, using new evidence or reasoning.
### Coordination labels (attempts to promote understanding and consensus):
#### Label A-Bailing out: An indication that a person is giving up on a discussion and will no longer engage.
#### Label B-Contextualisation: Usually in the first utterance, an individual “sets the stage" by describing what aspect they are challenging.
#### Label C-Asking questions: Seeking to understand another person’s opinion better. This does not include rhetorical questions, which are generally disagreement moves.
#### Label D-Providing clarification: Answering questions or providing information which seeks to create understanding, rather than only furthering a point.
#### Label E-Suggesting a compromise: An attempt to find a midway between one’s own point and the opposer’s.
#### Label F-Coordinating: In disagreement threads, discussions about edits that may indicate a compromise.
#### Label G-Conceding/recanting: An explicit admission that an interlocutor is willing to relinquish their point.
#### Label H-I don't know: Admitting that one is uncertain-a signal that the speaker is receptive to the idea that there are unknowns which may impact their argument.
#### Label I-Other: For utterances not covered by any other class, for instance, social niceties. 




```python
from DiscQuA import calculate_dispute_tactics

    calculate_dispute_tactics(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )

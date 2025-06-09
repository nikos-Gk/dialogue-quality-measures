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

```python
from DiscQuA import convert_csv_to_json
convert_csv_to_json(
    input_file=csv_path
)
```

disc_directory: directory with json files


## structure features

```python
from DiscQuA import calculate_structure_features
 calculate_structure_features(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## turn taking

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

```python
from DiscQuA import calculate_language_features

     calculate_language_features(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## controversy

```python
from DiscQuA import calculate_controversy

    calculate_controversy(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

## coherence

### coherece conversation with openai

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

### coherece conversation with llama, local inference. Using llama-cpp-python that supports gguf models

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


### coherece response 

```python
from DiscQuA import calculate_coherece_response

       calculate_coherece_response(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

### coherece ecoh 

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


## power of status and social bias

###power of status 

```python
from DiscQuA import calculate_coordination_per_discussion

       calculate_coordination_per_discussion(
        input_directory=disc_directory,
        moderator_flag=False,
    )
```

###social bias

```python
from DiscQuA import calculate_social_bias

    calculate_social_bias(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )
```

## politeness

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

```python
from DiscQuA import calculate_arg_dim

    calculate_arg_dim(
        input_directory=disc_directory,
        openAIKEY="", 
        moderator_flag=False
    )

```

## Overall Argument Quality

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

```python
from DiscQuA import calculate_dispute_tactics

    calculate_dispute_tactics(
        input_directory=disc_directory,
        openAIKEY="",
        moderator_flag=False,
    )

import re

from fastembed.rerank.cross_encoder import TextCrossEncoder

from sense_inventory import wordnet_labels


model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


def choose_sense(context, definitions):
    if len(definitions) == 1:
        return next(iter(definitions))

    scores = model.rerank(context, list(definitions.values()))
    scores_by_key = dict(zip(definitions, scores))
    return max(scores_by_key, key=scores_by_key.get)


def predict_with_reranker(sentence, target):
    labels_by_sense = wordnet_labels(target)
    definitions = {
        label: sense.definition()
        for sense, label in labels_by_sense.items()
    }
    return choose_sense(sentence, definitions)


def predict_with_window(sentence, target):
    words = list(re.finditer(r"\b\w+\b", sentence))
    for index, word in enumerate(words):
        if word.group().lower() == target:
            first_word = words[max(0, index - 4)]
            last_word = words[min(len(words) - 1, index + 4)]
            context = sentence[first_word.start() : last_word.end()]
            return predict_with_reranker(context, target)
    return "unknown"
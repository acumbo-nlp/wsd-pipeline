import re
from fastembed.rerank.cross_encoder import TextCrossEncoder

from lesk_baseline import labels_by_sense


model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
labels = list(labels_by_sense.values())
definitions = [sense.definition() for sense in labels_by_sense]


def predict_bank_with_reranker(sentence):
    scores = dict(zip(labels, model.rerank(sentence, definitions)))
    return max(scores, key=scores.get)


def predict_bank_with_window(sentence):
    words = list(re.finditer(r"\b\w+\b", sentence))
    for index, word in enumerate(words):
        if word.group().lower() == "bank":
            first_word = words[max(0, index - 4)]
            last_word = words[min(len(words) - 1, index + 4)]
            context = sentence[first_word.start() : last_word.end()]
            return predict_bank_with_reranker(context)
    return "unknown"
from nltk.tokenize import wordpunct_tokenize
from nltk.wsd import lesk

from sense_inventory import wordnet_labels


def predict_with_lesk(sentence, target):
    labels_by_sense = wordnet_labels(target)
    words = wordpunct_tokenize(sentence.lower())
    chosen_sense = lesk(words, target, synsets=list(labels_by_sense))
    return labels_by_sense[chosen_sense]
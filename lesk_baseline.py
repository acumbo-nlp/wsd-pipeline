from nltk.corpus import wordnet
from nltk.tokenize import wordpunct_tokenize
from nltk.wsd import lesk


labels_by_sense = {
    wordnet.synset("bank.n.01"): "river bank",
    wordnet.synset("depository_financial_institution.n.01"): "financial institution",
}


def predict_bank_with_lesk(sentence):
    words = wordpunct_tokenize(sentence.lower())
    chosen_sense = lesk(words, "bank", synsets=list(labels_by_sense))
    return labels_by_sense[chosen_sense]
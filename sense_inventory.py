SENSES_BY_TARGET = {
    "bank": {
        "bank.n.01": "river bank",
        "depository_financial_institution.n.01": "financial institution",
    },
    "plant": {
        "plant.n.01": "factory",
        "plant.n.02": "living plant",
    },
    "bat": {
        "bat.n.01": "flying mammal",
        "bat.n.05": "sports bat",
    },
    "crane": {
        "crane.n.04": "lifting machine",
        "crane.n.05": "bird",
    },
    "mouse": {
        "mouse.n.01": "rodent",
        "mouse.n.04": "computer mouse",
    },
    "spring": {
        "spring.n.01": "season",
        "spring.n.02": "mechanical spring",
    },
}


def wordnet_labels(target):
    from nltk.corpus import wordnet

    labels = {}
    for sense_id, label in SENSES_BY_TARGET[target].items():
        labels[wordnet.synset(sense_id)] = label
    return labels
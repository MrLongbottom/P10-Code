import pickle


class Model:
    def __init__(self, num_topics, alpha, eta, doc_topic, topic_word, name):
        self.num_topics = num_topics
        self.alpha = alpha
        self.eta = eta
        self.doc_topic = doc_topic
        self.topic_word = topic_word
        self.name = name

    def to_dict(self):
        return {"num_topic": self.num_topics,
                "alpha": self.alpha,
                "eta": self.eta,
                "doc_topic": self.doc_topic,
                "topic_word": self.topic_word}

    def to_str(self):
        return f"{self.num_topics}_{self.alpha}_{self.eta}_{self.name}"

    def save_model(self, save_path=""):
        full_path = save_path + self.to_str()
        with open(full_path, "wb") as file_path:
            pickle.dump(self.to_dict(), file_path)


def load_model(load_path):
    with open(load_path, 'rb') as file_path:
        return pickle.load(file_path)

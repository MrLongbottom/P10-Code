import pickle


class Model:
    def __init__(self, num_topics, alpha, eta, doc_topic, topic_word, doc_topic_count, topic_word_count, name):
        self.num_topics = num_topics
        self.alpha = alpha
        self.eta = eta
        self.doc_topic = doc_topic
        self.topic_word = topic_word
        self.doc_topic_count = doc_topic_count
        self.topic_word_count = topic_word_count
        self.name = name

    def to_dict(self):
        return {"num_topic": self.num_topics,
                "alpha": self.alpha,
                "eta": self.eta,
                "doc_topic": self.doc_topic,
                "topic_word": self.topic_word,
                "doc_topic_count": self.doc_topic_count,
                "topic_word_count": self.topic_word_count}

    def to_str(self):
        return f"{self.num_topics}_{self.alpha}_{self.eta}_{self.name}"

    def save_model(self, save_path=""):
        full_path = save_path + self.to_str()
        with open(full_path, "wb") as file_path:
            pickle.dump(self.to_dict(), file_path)


def load_model(load_path):
    with open(load_path, 'rb') as file_path:
        dict_model = pickle.load(file_path)
        return Model(dict_model["num_topic"],
                     dict_model["alpha"],
                     dict_model["eta"],
                     dict_model["doc_topic"],
                     dict_model["topic_word"],
                     dict_model["doc_topic_count"],
                     dict_model["topic_word_count"],
                     file_path.name.split("_")[-1])


def wrong_save_fix(load_path: str, save_path: str = ""):
    wrong_model = load_model(load_path)
    correct_model = Model(wrong_model.num_topics, wrong_model.alpha, wrong_model.eta, wrong_model.doc_topic,
                          wrong_model.doc_topic_count, wrong_model.topic_word, wrong_model.topic_word_count,
                          wrong_model.name)
    correct_model.save_model(save_path)


if __name__ == '__main__':
    wrong_save_fix(load_path="models/90_0.01_0.1_author", save_path="models/")

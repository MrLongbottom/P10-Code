import pickle

from ray import tune
import os
import pandas as pd

from gibbs_utility import get_coherence
from model.gibbs_lda import gibbs_sampling, setup
from preprocess.preprocessing import prepro_file_load


def training_function(config):
    # Hyperparameters
    alpha, beta, num_topics = config["alpha"], config["beta"], config["num_topics"]
    print("got to training")
    train_docs, test_docs, \
    word_topic_assignment, document_topic_dist, topic_word_dist, \
    word_topic_count, doc_topic_count = setup(alpha=alpha, beta=beta, num_topics=num_topics)

    
    for step in range(50):
        print(f"step: {step}")
        # Iterative training function - can be any arbitrary training procedure.
        gibbs_sampling(train_docs,
                       document_topic_dist, topic_word_dist,
                       word_topic_count, doc_topic_count,
                       word_topic_assignment)
        coherence_score = get_coherence(doc2bow, corpora, texts, corpora, num_topics, topic_word_dist)

        tune.report(topic_coherence=coherence_score)


if __name__ == '__main__':  
    print(os.getcwd())
    doc2bow, corpora, texts = prepro_file_load('doc2bow'), \
                              prepro_file_load('corpora'), \
                              list(prepro_file_load('doc2pre_text').values())
    
    analysis = tune.run(
        training_function,
        config={
            "alpha": tune.choice([0.001, 0.01, 0.1]),
            "beta": tune.choice([0.01, 0.1]),
            "num_topics": tune.grid_search([50, 60, 70, 80])
        })

    print("Best config: ", analysis.get_best_config(metric="topic_coherence", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    
    pd.to_pickle("result_df.pkl")

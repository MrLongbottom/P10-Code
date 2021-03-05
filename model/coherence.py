from gensim.models.coherencemodel import CoherenceModel


# supported measures = {'u_mass', 'c_v', 'c_uci', 'c_npmi'}
def coherence(topics, doc2bow, dictionary, texts, coherence_measure = 'c_v'):
    cm = CoherenceModel(topics=topics, corpus=doc2bow, dictionary=dictionary, texts=texts, coherence=coherence_measure)
    return cm.get_coherence()

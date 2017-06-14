import numpy as np

def random_digit(doc_labels):
    return np.random.randint(10, size=(doc_labels.shape[0], 1))

def random_from_docs(doc_labels):
    indices = np.random.randint(doc_labels.shape[1], size=doc_labels.shape[0])
    indices = np.vstack([np.arange(doc_labels.shape[0]), indices]).T
    return np.reshape(doc_labels[indices[:, 0], indices[:, 1]], (doc_labels.shape[0], 1))

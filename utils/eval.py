import numpy as np

from search.reward import ndcg_full

# Calculates the NDCG @ list for all possible queries and the
# classification accuracy of the model.
def evaluate_model(model, evaluation_set, sess, num_queries):
        images = evaluation_set.images
        labels = evaluation_set.labels

        # Prepare the input.
        x = np.reshape(images, (1, images.shape[0], images.shape[1]))
        labels = np.reshape(labels, (1, len(labels)))

        # Calculate the NDCG for each possible query.
        ndcgs = np.zeros(num_queries)
        for query_id in range(num_queries):

            # Create the query input.
            queries = np.array([[query_id]])

            # Retrieve deterministic ranking for the query, sorted by document score.
            feed_dict = { model.x: x, model.q: queries }
            ranking = sess.run(model.det_ranking, feed_dict=feed_dict)

            # Calculate the ndcg for this list.
            rel_labels = np.zeros(labels.shape)
            rel_labels[np.where(labels == queries)] = 1.
            ndcgs[query_id] = ndcg_full(ranking, rel_labels)[0, 0]

        # Calculate the accuracy of the model by classifying documents as the output
        # with the highest document score.
        feed_dict = { model.x: x, model.true_labels: labels }
        accuracy = sess.run(model.accuracy, feed_dict=feed_dict)[0]

        return accuracy, ndcgs

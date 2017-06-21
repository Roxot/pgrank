import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

def write_evaluation_summaries(summ_writer, iteration, accuracy, average_ndcg):
    ndcg_summary = tf.Summary(value=[tf.Summary.Value(tag="average_ndcg", \
            simple_value=float(average_ndcg))])
    acc_summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy", \
            simple_value=float(accuracy))])
    summ_writer.add_summary(acc_summary, iteration)
    summ_writer.add_summary(ndcg_summary, iteration)

def write_train_summaries(summ_writer, iteration, batch_reward, loss):
    reward_summary = tf.Summary(value=[tf.Summary.Value(tag="batch_reward", \
            simple_value=float(batch_reward))])
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", \
            simple_value=float(loss))])
    summ_writer.add_summary(reward_summary, iteration)
    summ_writer.add_summary(loss_summary, iteration)

def mnist_image_summary(model, evaluation_set, sess):
    images = evaluation_set.images
    labels = evaluation_set.labels

    # Prepare the input.
    x = np.reshape(images, (1, images.shape[0], images.shape[1]))
    labels = np.reshape(labels, (1, len(labels)))

    # Retrieve the top 10 images for each query.
    for query_id in range(10):

        # Create the query input.
        queries = np.array([[query_id]])

        # Retrieve deterministic ranking for the query, sorted by document score.
        feed_dict = { model.x: x, model.q: queries }
        ranking = sess.run(model.det_ranking, feed_dict=feed_dict)

        top_10 = images[ranking[0, :10]]
        for rank, image in enumerate(top_10):
            plt.subplot(10, 10, (10 * rank) + 1 + query_id)
            if rank == 0:
                plt.title("%d" % query_id)
            plt.axis('off')
            plt.imshow(image.reshape(28, 28), cmap="gray")

    plt.suptitle('Top 10 images for different queries')

    # Create a TensorFlow summary of the matplotlib plot.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return tf.image_summary("top_10_images", image)

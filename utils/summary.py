import tensorflow as tf

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

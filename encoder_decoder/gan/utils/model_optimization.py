import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def create_dis_pretrain_op(dis_loss, buckets):
    with tf.name_scope('pretrain_discriminator'):
        optimizer = tf.train.AdamOptimizer(FLAGS.dis_pretrain_learning_rate)
        dis_vars = [
            v for v in tf.trainable_variables() if v.op.name.startswith('dis')
        ]
        # don't update share embedding
        gradient_norms = []
        updates = []
        for bucket_id, _ in enumerate(buckets):
            dis_grads = tf.gradients(dis_loss, dis_vars)
            dis_grads_clipped, norm = tf.clip_by_global_norm(
                dis_grads, FLAGS.max_gradient_norm)
            gradient_norms.append(norm)
            dis_pretrain_op = optimizer.apply_gradients(
                zip(dis_grads_clipped, dis_vars))
            updates.append(dis_pretrain_op)
        return gradient_norms, updates


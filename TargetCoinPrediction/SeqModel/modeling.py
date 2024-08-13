# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import gc

# Configuration for memory management
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

class ModelTraining:
    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        self.accumulation_steps = 4  # Number of steps to accumulate gradients
        self.global_step = 0

    def loss_op(self, sess):
        try:
            with tf.name_scope('Metrics'):
                # Reshape tensors to ensure they match
                original_label_shape = tf.shape(self.label)
                original_logit_shape = tf.shape(self.logit)

                # Reshape to the desired shape (256 in this case)
                self.label = tf.reshape(self.label, [-1, 256])
                self.logit = tf.reshape(self.logit, [-1, 256])

                # Calculate loss
                loss = sess.run(self.model.loss)

                # Initialize the optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"])

                # Compute gradients
                gradients, variables = zip(*optimizer.compute_gradients(loss))

                # Clip gradients to prevent explosion
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

                # Accumulate gradients
                if self.global_step % self.accumulation_steps == 0:
                    self.accumulated_gradients = [grad for grad in clipped_gradients]
                else:
                    self.accumulated_gradients = [accum + grad for accum, grad in zip(self.accumulated_gradients, clipped_gradients)]

                # Apply gradients every `accumulation_steps` steps
                if (self.global_step + 1) % self.accumulation_steps == 0:
                    sess.run(optimizer.apply_gradients(zip(self.accumulated_gradients, variables)))
                    self.accumulated_gradients = None

                # Revert tensors to their original shape
                self.label = tf.reshape(self.label, original_label_shape)
                self.logit = tf.reshape(self.logit, original_logit_shape)

                # Increment global step
                self.global_step += 1

            return loss

        except tf.errors.InvalidArgumentError as e:
            print(f"Shape mismatch error at step {self.global_step}: {e}")
            return None

    def train(self, sess, dataset):
        for batch in dataset:
            self.label = batch['label']
            self.logit = self.model(batch['features'])
            
            # Call the loss operation to accumulate or apply gradients
            loss = self.loss_op(sess)
            
            if loss is not None:
                # Optional: Log the loss if needed
                print(f'Loss at step {self.global_step}: {loss}')
            
            # Force garbage collection to manage memory
            gc.collect()

# Usage Example
model_config = {"learning_rate": 1e-5}
# Assuming `model` and `dataset` are properly initialized
# training = ModelTraining(model, model_config=model_config)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     training.train(sess, dataset)

class DNN(object):
    def __init__(self, tensor_dict, model_config):
        self.label = tensor_dict["label"]
        self.channel_embedding = tensor_dict["channel_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]
        self.target_features = tensor_dict["target_features"]

        # Configuration
        self.model_config = {
            "hidden1": 32,
            "hidden2": 16,
            "hidden3": 32,
            "learning_rate": 0.0005
        }
        self.model_config.update(model_config)
        print("Model Configuration:")
        print(self.model_config)

    def build(self):
        """
        Build the architecture for the base DNN model.
        """
        self.inp = self.target_features
        self.build_fcn_net(self.inp)
        self.loss_op()

    def build_fcn_net(self, inp):
        with tf.name_scope("Fully_connected_layer"):
            dnn1 = tf.layers.dense(inp, self.model_config["hidden1"], activation=tf.nn.relu, name='f1')
            dnn2 = tf.layers.dense(dnn1, self.model_config["hidden2"], activation=tf.nn.relu, name='f2')
            self.logit = tf.squeeze(tf.layers.dense(dnn2, 1, activation=None, name='logit'))

        self.y_hat = tf.sigmoid(self.logit)

    def loss_op(self):
        with tf.name_scope('Metrics'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"])
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
            self.optimizer = optimizer.apply_gradients(zip(clipped_gradients, variables))

class SNN(DNN):
    def __init__(self, tensor_dict, model_config):
        super(SNN, self).__init__(tensor_dict, model_config)
        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:,:,-9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]

    def positional_attention_layer(self):
        feat_num = self.seq_embedding.shape[2]
        self.length = tf.where(tf.less_equal(self.length, self.model_config["max_seq_length"]),
                               self.length, tf.constant(self.model_config["max_seq_length"], shape=[self.model_config["batch_size"]]))

        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        mask_2d = tf.expand_dims(self.sequence_mask, axis=2)
        masked_seq_embedding = self.seq_embedding * tf.cast(mask_2d, tf.float32)

        pos_atten_list = []
        seq_pos_attent_embed_list = []
        num_head = 8

        for i in range(feat_num):
            x = tf.expand_dims(masked_seq_embedding[:, :, i], axis=2)
            inp = layer_norm(x, name="layer_norm_" + str(i))

            pos_atten = tf.get_variable("pos_atten_" + str(i), [1, 50, num_head],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
            pos_atten = tf.layers.dense(pos_atten, 8, activation=tf.nn.relu, name="pos_" + str(i) + "_fx1")
            pos_atten = tf.layers.dense(pos_atten, num_head, activation=None, name="pos_" + str(i) + "_fx2")

            pos_atten = tf.tile(pos_atten, [self.model_config["batch_size"], 1, 1])
            paddings = tf.ones_like(pos_atten) * (-2 ** 32 + 1)

            mask_new = tf.tile(mask_2d, [1, 1, num_head])
            pos_atten = tf.where(tf.logical_not(mask_new), paddings, pos_atten)
            pos_atten = tf.nn.softmax(pos_atten, axis=1)

            seq_pos_attent_embed = tf.reduce_sum(pos_atten * inp, axis=1)
            seq_pos_attent_embed_list.append(seq_pos_attent_embed)
            pos_atten_list.append(tf.expand_dims(pos_atten, axis=2))

        self.pos_atten = tf.concat(pos_atten_list, axis=2)
        self.seq_pos_attent_embeding = tf.concat(seq_pos_attent_embed_list, axis=1)

        coin_mask_2d = tf.expand_dims(self.sequence_mask, axis=2)
        masked_seq_coin_embedding = self.seq_coin_embedding * tf.cast(coin_mask_2d, tf.float32)
        coin_inp = layer_norm(masked_seq_coin_embedding, name="layer_norm_coin_embedding")

        pos_atten_coin = tf.get_variable("pos_atten_coin", [1, 50, num_head],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

        pos_atten_coin = tf.layers.dense(pos_atten_coin, num_head, activation=tf.nn.relu, name="pos_coin_fx1")
        pos_atten_coin = tf.layers.dense(pos_atten_coin, num_head, activation=None, name="pos_coin_fx2")

        pos_atten_coin = tf.tile(pos_atten_coin, [self.model_config["batch_size"], 1, 1])
        paddings = tf.ones_like(pos_atten_coin) * (-2 ** 32 + 1)

        coin_mask_new = tf.tile(coin_mask_2d, [1, 1, num_head])
        pos_atten_coin = tf.where(tf.logical_not(coin_mask_new), paddings, pos_atten_coin)
        pos_atten_coin = tf.nn.softmax(pos_atten_coin, axis=1)

        seq_coin_pos_attent_embeding = tf.reduce_sum(tf.expand_dims(pos_atten_coin, axis=2) * tf.expand_dims(coin_inp, axis=3), axis=1)
        self.seq_coin_pos_attent_embeding = tf.reshape(seq_coin_pos_attent_embeding, [self.model_config["batch_size"], -1])
        output = tf.concat([self.seq_pos_attent_embeding, self.seq_coin_pos_attent_embeding], axis=1)

        return output

    def build(self):
        self.seq_embedding_mean = self.positional_attention_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()

class SNNTA(DNN):
    def __init__(self, tensor_dict, model_config):
        super(SNNTA, self).__init__(tensor_dict, model_config)
        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:,:,-9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]
        self.atten_strategy = "mlp"

    def positional_attention_layer(self):
        self.length = tf.where(tf.less_equal(self.length, self.model_config["max_seq_length"]), self.length,
                               tf.constant(self.model_config["max_seq_length"], shape=[self.model_config["batch_size"]]))
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        query = self.target_features[:,-9:]
        key = self.seq_embedding

        if self.atten_strategy == "mlp":
            query = tf.expand_dims(query, axis=1)
            queries = tf.tile(query, [1, key.shape[1], 1])
            attention_all = tf.concat([queries, key, queries - key, queries * key], axis=-1)

            d_layer_1_all = tf.layers.dense(attention_all, 32, activation=tf.nn.relu, name="f1_att")
            d_layer_2_all = tf.layers.dense(d_layer_1_all, 16, activation=tf.nn.relu, name="f2_att")
            d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att")
            d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, key.shape[1]])

            scores = d_layer_3_all
            key_masks = tf.expand_dims(self.sequence_mask, 1)

            paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
            scores = tf.where(key_masks, scores, paddings)
            scores = tf.nn.softmax(scores, axis=2)
            output = tf.squeeze(tf.matmul(scores, key))

        elif self.atten_strategy == "dot":
            attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                     keys=key,
                                                     values=key,
                                                     num_heads=8,
                                                     key_masks=self.sequence_mask,
                                                     dropout_rate=self.model_config["dropout_rate"],
                                                     is_training=self.model_config["is_training"],
                                                     reuse=False)
            output = tf.squeeze(attended_embedding, axis=1)

        else:
            raise ValueError("Please choose the right attention strategy. Current is " + self.atten_strategy)

        return layer_norm(output, "layer_norm")

    def build(self):
        self.seq_embedding_mean = self.positional_attention_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    epsilon = 1e-6
    filters = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        scale = tf.get_variable("layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [filters], initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(input_tensor, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=-1, keep_dims=True)
        input_tensor = (input_tensor - mean) * tf.rsqrt(variance + epsilon)
        input_tensor = input_tensor * scale + bias

        return input_tensor

def multihead_attention(queries,
                        keys,
                        values,
                        key_masks,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      values: A 3d tensor with shape of [N, T_v, C_v]
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.logical_not(key_masks), paddings, outputs)

        outputs = tf.nn.softmax(outputs)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries

    return outputs

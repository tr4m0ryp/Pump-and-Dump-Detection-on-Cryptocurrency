#-*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow v2 behavior for compatibility
from run_train import *    # Import functions and flags from the training script
import csv

def delete_flags(FLAGS, keys_list):
    print("Deleting specified flags...")
    for key in keys_list:
        if hasattr(FLAGS, key):
            try:
                FLAGS.__delattr__(key)
                print(f"Deleted flag: {key}")
            except Exception as e:
                print(f"Error deleting flag {key}: {str(e)}")

def redefine_flags():
    print("Redefining flags for evaluation...")
    tf.flags.DEFINE_bool("do_train", False, "Set to False for evaluation.")
    tf.flags.DEFINE_bool("do_eval", True, "Set to True for evaluation.")
    tf.flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint for evaluation.")
    tf.flags.DEFINE_integer("init_seed", 1234, "Random seed for initialization.")
    print("Flags redefined.")

def check_tensor_shapes(tensor_dict):
    print("Checking tensor shapes...")
    for name, tensor in tensor_dict.items():
        print(f"Tensor '{name}' shape: {tensor.shape}")

def validate_tensors(tensor_dict):
    """Validates tensor shapes and data types in the provided dictionary."""
    print("Validating tensor shapes and data types...")
    reference_shape = None
    for key, tensor in tensor_dict.items():
        if reference_shape is None:
            reference_shape = tensor.shape
        elif tensor.shape[0] != reference_shape[0]:
            raise ValueError(f"Tensor {key} has inconsistent batch size {tensor.shape[0]} compared to {reference_shape[0]}")
        if tensor.dtype != tf.float32:
            print(f"Warning: Tensor '{key}' is {tensor.dtype}, expected float32. Casting...")
            tensor_dict[key] = tf.cast(tensor, tf.float32)

def setup_tf_dataset(tensor_dict, batch_size):
    print("Setting up TensorFlow dataset...")
    try:
        validate_tensors(tensor_dict)  # Validate tensor consistency
        dataset = tf.data.Dataset.from_tensor_slices(tensor_dict)
        print(f"Dataset element spec: {dataset.element_spec}")
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        print("Dataset shuffled.")
        dataset = dataset.batch(batch_size)
        print(f"Dataset batched with batch size: {batch_size}")
        dataset = dataset.repeat()  # Ensure the dataset repeats indefinitely
        print("Dataset set to repeat.")
        return dataset
    except Exception as e:
        print(f"Error setting up TensorFlow dataset: {str(e)}")
        return None

def evaluate_model(session, dataset, metrics_op, num_epochs=1):
    print("Starting model evaluation...")
    try:
        session.run(tf.local_variables_initializer())
        print("Local variables initialized.")

        # Create the dataset iterator and initialize it
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        next_element = iterator.get_next()

        pre_probas = []
        coin_ids = []
        labels = []

        for epoch_num in range(1, num_epochs + 1):
            session.run(iterator.initializer)  # Initialize the iterator for each epoch
            print(f"Running session for epoch {epoch_num}...")

            while True:
                try:
                    # Fetch the next batch
                    input_data, label, coin_id = session.run([next_element['input_tensor'], next_element['label_tensor'], next_element['coin_id']])
                    print(f"Batch data - input_data shape: {input_data.shape}, labels shape: {label.shape}")

                    # Create feed_dict for model evaluation
                    feed_dict = {'input_tensor:0': input_data, 'label_tensor:0': label}
                    accuracy, auc = session.run([metrics_op['accuracy'], metrics_op['auc']], feed_dict=feed_dict)

                    pre_probas += list(accuracy)
                    labels += list(label)
                    coin_ids += list(coin_id)

                    # Print the metrics for the current batch
                    print(f"Epoch {epoch_num}, Batch Accuracy: {accuracy}, Batch AUC: {auc}")

                except tf.errors.OutOfRangeError:
                    print(f"End of epoch {epoch_num}.")
                    break
        
        # Writing the predictions to a CSV file
        with open('predicted_coins.csv', 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'coin_id', 'predicted_probability']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # Write the header row
            for coin_id, prob in zip(coin_ids, pre_probas):
                writer.writerow({'epoch': epoch_num, 'coin_id': coin_id, 'predicted_probability': prob})

        print("Predicted coin probabilities saved to predicted_coins.csv")

        return pre_probas, labels
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        return [], []

if __name__ == '__main__':
    print("Starting script...")
    delete_flags(tf.flags.FLAGS, ["do_train", "do_eval", "init_checkpoint", "init_seed"])
    redefine_flags()

    # TensorDict setup
    tensor_dict = {
        'input_tensor': tf.placeholder(tf.float32, shape=[None, 256]),  # Replace with actual data
        'label_tensor': tf.placeholder(tf.float32, shape=[None, 1]),     # Replace with actual data
        'coin_id': tf.placeholder(tf.float32, shape=[None])              # Replace with actual coin ID tensor
    }

    check_tensor_shapes(tensor_dict)

    batch_size = 256
    dataset = setup_tf_dataset(tensor_dict, batch_size)

    if dataset is None:
        print("Dataset setup failed. Exiting.")
        exit()

    metrics_op = {
        'accuracy': tf.metrics.accuracy(labels=tf.placeholder(tf.float32, shape=[None]), predictions=tf.placeholder(tf.float32, shape=[None])),
        'auc': tf.metrics.auc(labels=tf.placeholder(tf.float32, shape=[None]), predictions=tf.placeholder(tf.float32, shape=[None]))
    }

    with tf.Session() as sess:
        print("TensorFlow session started.")
        if tf.flags.FLAGS.init_checkpoint:
            print(f"Restoring model from checkpoint: {tf.flags.FLAGS.init_checkpoint}")
            saver = tf.train.Saver()
            saver.restore(sess, tf.flags.FLAGS.init_checkpoint)
            print("Model restored.")

        # You can set the number of epochs here
        num_epochs = 5  # Set this to the number of epochs you want
        pre_probas, labels = evaluate_model(sess, dataset, metrics_op, num_epochs=num_epochs)

        try:
            tf.app.run()
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")

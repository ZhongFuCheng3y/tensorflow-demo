import tensorflow as tf


def _decode(example, type):
    features_config = {}

    features_config['item_id'] = tf.FixedLenFeature([1], tf.int64)
    features = tf.parse_example(example, features=features_config)

    itemid = tf.cast(features['item_id'], tf.int32)

    return itemid


def batch_inputs(files, batch_size, type, num_epochs=None, num_preprocess_threads=2):
    """Reads input data num_epochs times.
    """""
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        dataset = tf.data.Dataset.from_tensor_slices(files).interleave(
            lambda x: tf.data.TFRecordDataset(x).prefetch(10), cycle_length=num_preprocess_threads)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.map(lambda x: _decode(x, type), num_parallel_calls=2)
        dataset = dataset.shuffle(buffer_size=batch_size * 2)
        dataset = dataset.prefetch(buffer_size=20)
        return dataset


def read_tensorflow_tfrecord_files():
    # 开始定义dataset以及解析tfrecord格式
    train_filenames = tf.placeholder(tf.string, shape=[None])
    vali_filenames = tf.placeholder(tf.string, shape=[None])

    # 加载train_dataset
    train_dataset = batch_inputs([
        train_filenames], batch_size=5, type=False,
        num_epochs=2, num_preprocess_threads=3)
    # 加载validation_dataset
    validation_dataset = batch_inputs([vali_filenames
                                       ], batch_size=5, type=False,
                                      num_epochs=2, num_preprocess_threads=3)

    # 创建出string_handler()的迭代器（通过相同数据结构的dataset来构建)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    # 有了迭代器就可以调用next方法了。
    itemid = iterator.get_next()

    # 指定哪种具体的迭代器，有单次迭代的，有初始化的。
    training_iterator = train_dataset.make_initializable_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    # 定义出placeholder的值
    training_filenames = [
        "/Users/zhongfucheng/tfrecord_test/data01aa"]
    validation_filenames = ["/Users/zhongfucheng/tfrecord_validation/part-r-00766"]

    with tf.Session() as sess:
        # 初始化迭代器
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        print(training_handle)
        print(validation_handle)

        for _ in range(2):
            # Run 200 steps using the training dataset. Note that the training dataset is
            # infinite, and we resume from where we left off in the previous `while` loop
            # iteration.
            sess.run(training_iterator.initializer, feed_dict={train_filenames: training_filenames})
            print("this is training iterator ----")

            for _ in range(5):
                print(sess.run(itemid, feed_dict={handle: training_handle}))

            sess.run(validation_iterator.initializer,
                     feed_dict={vali_filenames: validation_filenames})

            print("this is validation iterator ")
            for _ in range(5):
                print(sess.run(itemid, feed_dict={vali_filenames: validation_filenames, handle: validation_handle}))


if __name__ == '__main__':
    read_tensorflow_tfrecord_files()

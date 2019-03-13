import tensorflow as tf
import time
import pandas as pd
import numpy as np

job_name = "worker"
task_id = 0

strworker_hosts = "stghislain:2222,london:2222,frankfurt:2222,eemshaven:2222"
strps_hosts = "oregon:2222"

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
TRAINING_STEPS = 100
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
MODEL_SAVE_PATH = "./log"


def creatDataSet():
    trainDataSet = pd.read_csv("./data/data0.csv")
    trainDataSet = trainDataSet.as_matrix()

    trainData = trainDataSet[:, 1:len(trainDataSet[0]) - 1]
    trainLabels = trainDataSet[:, len(trainDataSet[0]) - 1][:, np.newaxis]
    return trainData, trainLabels


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def build_model(x, y_, is_chief):
    l1 = add_layer(x, 79, 100, activation_function=tf.nn.sigmoid)
    l2 = add_layer(l1, 100, 150, activation_function=tf.nn.sigmoid)
    prediction = add_layer(l2, 150, 1, activation_function=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - prediction), reduction_indices=[1]))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()

    return global_step, loss, train_op

def main(argv=None):
    ps_hosts = strps_hosts.split(',')
    worker_hosts = strworker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_id)


    if job_name == 'ps':
        server.join()

    
    is_chief = (task_id == 0)

    with tf.device(
            tf.train.replica_device_setter(worker_device="/job:worker/task:%d " % task_id, cluster=cluster)):
        x = tf.placeholder(tf.float32, [None, 79])
        y_ = tf.placeholder(tf.float32, [None, 1])
        global_step, loss, train_op = build_model(x, y_, is_chief)
        saver = tf.train.Saver()
        
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(
            is_chief=is_chief,  
            logdir=MODEL_SAVE_PATH,  
            init_op=init_op, 
            summary_op=summary_op,  
            saver=saver,  
            global_step=global_step,  
            save_model_secs=60,  
            save_summaries_secs=60  
        )
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)
        
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        step = 0
        start_time = time.time()
        step_time = 0

        File = open('./timeLog_worker_' + str(task_id) + '.txt', "w")
        
        while not sv.should_stop():


            step_start_time = time.time()
            xs, ys = creatDataSet()
            _, loss_value, global_step_value = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if global_step_value >= TRAINING_STEPS: break
            
            if step > 0:
                duration = time.time() - start_time
                step_time = time.time() - step_start_time
                
                sec_per_batch = duration / global_step_value

                format_str = (
                    "After %d training steps (%d global steps), loss on training batch is %g. (%.3f sec/batch). This step use %.3f sec.")
                record = format_str % (step, global_step_value, loss_value, sec_per_batch, step_time)

                print(format_str % (step, global_step_value, loss_value, sec_per_batch, step_time))
                File.write(record + "\n")

            step += 1

        total_time = time.time() - start_time
        print("total time is %.3f sec." % total_time)
        record = "total time is %.3f sec." % total_time
        File.write(record + "\n")
        File.close()
        sv.stop()


if __name__ == "__main__":
    tf.app.run()

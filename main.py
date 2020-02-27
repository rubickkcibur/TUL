import tensorflow as tf
import BiLSTM
import util
import gc
import os
import numpy as np

learning_rate = 0.00095
train_iters = 50
dispaly_step = 1000
batch_size = 1
User_List = list()
keep_prob = tf.placeholder(tf.float32)
lambda_loss_amount = 0.001
embb_size = 256  # embedding size
hidden_size = 300  # hidden size
w_2 = 28
user_classes = 182  # clss number
n_steps = [3]
X = tf.placeholder("float", [batch_size, None, embb_size],name='X')
istate_fw = tf.placeholder("float", [None, 2 * hidden_size],name='fw')
istate_bw = tf.placeholder("float", [None, 2 * hidden_size],name='bw')
y_out = tf.placeholder("float", [batch_size, user_classes],name='y_out')
seq_length = tf.placeholder(tf.int32, [None],name='seq')

acc_list = []

weights = {
    'out': tf.Variable(tf.random_normal([2 * hidden_size, user_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([user_classes]))
}


pred = BiLSTM.RNN(X, weights, biases, keep_prob, hidden_size=hidden_size, seq_length=seq_length)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_out, logits=pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.argmax(y_out, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def getOneHot(i,classes):
    re = [0] * classes
    re[i] = 1
    return re

def test(sess, testT, testU, testL, allAcc, iter, filename='para_save/TEST_resultVec.txt'):  # Test
    ftestw = open(filename, 'a+')
    ftestw.write("iters:=" + str(iter) + " ")
    ftestw.write("allAcc:" + str(allAcc) + " ")

    step = 0
    Dic = {}
    for i in range(user_classes):
        Dic[i] = [0,0,0,0]  #
    while step < len(testT):  #
        user_id = testU[step]
        # print user_id
        Dic.get(user_id)[0] += 1  #total
        nowVec = sess.run(pred, feed_dict={X: [testT[step]], y_out: [getOneHot(testU[step],user_classes)],
                                               keep_prob: 0.5, seq_length: [testL[step]]})[0]  #y_out不用吧？
        top1 = np.argpartition(a=-nowVec, kth=1)[:1]
        top5 = np.argpartition(a=-nowVec, kth=5)[:5]
        top10 = np.argpartition(a=-nowVec, kth=10)[:10]
        if user_id == top1[0]:
            Dic.get(user_id)[1] += 1 #top1
        for i in range(5):
            if top5[i] == user_id:
                Dic.get(user_id)[2] += 1 #top5
                break
        for i in range(10):
            if top10[i] == user_id:
                Dic.get(user_id)[3] += 1 #top10
                break
        step += 1
    # Count Macro-F1
    total = 0
    top1 = 0
    top5 = 0
    top10 = 0
    ftestw.write("Dic:")
    for i in Dic.keys():
        ftestw.write(str(i) + ": " + str(Dic.get(i)[0]) + " " + str(Dic.get(i)[1]) +
                     " " + str(Dic.get(i)[2]) + " " + str(Dic.get(i)[3]))
        total += Dic.get(i)[0]
        top1 += Dic.get(i)[1]
        top5 += Dic.get(i)[2]
        top10 += Dic.get(i)[3]
    try:
        ftestw.write("OUT CONSOLE: ")

        ftestw.write("step=" + str(step) + " ")
        ftestw.write(" length=" + str(len(testT)) + " ")
        ftestw.write(" Accuracy1: " + str(top1/total) + " ")
        ftestw.write(" Accuracy5: " + str(top5/total) + " ")
        ftestw.write(" Accuracy10: " + str(top10 / total) + " ")
        ftestw.write('\n')
    except Exception:
        print('get error in count acc')
    return 0

def start():
    saver = tf.train.Saver()
    if not os.path.exists('para_save/'):
        os.mkdir('para_save/')
    with tf.Session() as sess:
        gc.collect()
        print("initializing...")
        sess.run(tf.initialize_all_variables())
        print('loading data...')
        trainU, trainT, trainL, testU, testT, testL = util.loaddata()
        print('data loaded')
        for i in range(train_iters):
            allAcc = 0
            acc = 0
            for step in range(len(trainT)):
                sess.run(optimizer, feed_dict={X: [trainT[step]], y_out: [getOneHot(trainU[step],user_classes)],
                                               keep_prob: 0.5, seq_length: [trainL[step]]})
                if (sess.run(correct_pred,
                             feed_dict={X: [trainT[step]], y_out: [getOneHot(trainU[step],user_classes)],
                                        keep_prob: 0.5, seq_length: [trainL[step]]}) == 1):
                    acc += 1
                if (step % dispaly_step == 0):
                    loss = sess.run(cost,
                                    feed_dict={X: [trainT[step]], y_out: [getOneHot(trainU[step],user_classes)],
                                        keep_prob: 0.5, seq_length: [trainL[step]]})
                    print('step=', step, ' cost=', loss)
                    print("Iter " + str(i) + ", MiniNowItem Loss= " + \
                          "{:.6f}".format(loss) \
                          + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                    allAcc += acc
                    acc = 0
            print("Iter No:", i, "Finished! allAcc=", allAcc)
            acc_list.append(allAcc / len(trainT))
            saver.save(sess, './para_save/temp_rnn.pkt')
            test(sess=sess,testT=testT,testU=testU,testL=testL,allAcc=allAcc,iter=i)
        print("All Optimization Finished!")
        fresultw = open('para_save/result_Weights.txt', 'w')
        fresultb = open('para_save/result_Biases.txt', 'w')
        rweights = sess.run(weights['out'])
        rbiases = sess.run(biases['out'])
        for i in range(hidden_size):  #
            for j in range(user_classes):
                fresultw.write('%f ' % rweights[i][j])
            fresultw.write('\n')

        for i in range(user_classes):  #
            fresultb.write('%f ' % rbiases[i])
        fresultw.close()
        fresultb.close()
        print("iter:" + str(iter) + "done")

if __name__ == "__main__":
    start()
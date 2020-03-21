import tensorflow.compat.v1 as tf #텐서플로우 1 버젼
tf.disable_v2_behavior() #버젼 2를 안쓰겠다
import numpy as np
"""
[[0,0], -> [1, 0, 0] 기타
 [1,0], -> [0, 1, 0] 포유류
 [1,1], -> [0, 0, 1] 조류
 [0,0], -> [1, 0, 0] 기타
 [0,0], -> [1, 0, 0] 기타
 [0,1]  -> [0, 0, 1] 조류
"""
class Mammal:
    @staticmethod
    def execute():
        # [털, 날개] -> 기타, 포유류, 조류
        x_data = np.array(
            [[0, 0],
             [1, 0],
             [1, 1],
             [0, 0],
             [0, 0],
             [0, 1]
             ]
        )
        y_data = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]
             ]
        )
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_uniform([2,3],-1,1.))
        # -1 all
        # nn dms 2차원으로 [입력층(특성), 출력층(레이블)] -> [2,3]
        #nn이 tf안에 있다
        b = tf.Variable(tf.zeros([3]))
        #b는 평향 bias
        #b는 각 레이어의 아웃풋 갯수로 결정함
        L = tf.add(tf.matmul(X,W), b)  # matmul : 곱하기 l = WX + b
        L = tf.nn.relu(L)
        model = tf.nn.softmax(L)

        """
        softmax함수는 다음처럼 결과값을 전체 합이 1인 확률로 만들어 주는 함수
        예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23] : scale
        """
        print(f'모델 내부 보기 {model}')
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model),axis=1 )) #cost : 차이
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #경사하강법
        train_op = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for step in range(100):
            sess.run(train_op, {X:x_data, Y : y_data})
            if (step + 1) ==0 :
                print( step +1, sess.run(cost, { X : x_data, Y : y_data}))
        #결과 확인
        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y, 1)
        print(f' 예측값 {sess.run(prediction, {X: x_data})}')
        print(f' 실제값 {target, {Y: y_data}}')
        is_connect = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_connect, tf.float32))
        print(' 정확도 : %.2f' % sess.run(accuracy * 100, {X:x_data, Y:y_data}))

if __name__ == '__main__':
    Mammal.execute()
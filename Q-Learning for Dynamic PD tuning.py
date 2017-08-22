
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy.matlib


# In[2]:

#Timing Law

def s(t): return -4*np.pi*(t/T)**3 + 6*np.pi*(t/T)**2
def s_dot(t): return (-12*np.pi*(t/T)**2 + 12*np.pi*t/T)/T
def s_2dot(t): return (-24*np.pi*t/T + 12*np.pi)/(T**2)

#Trajectory
r = np.pi;
def q_d(t): return np.array([r*np.cos(s(t)),r*np.sin(s(t))])
def q_d_dot(t): return np.array([-r*np.sin(s(t))*s_dot(t), r*np.cos(s(t))*s_dot(t)])
def q_2d_dot(t): return np.array([-r*np.cos(s(t))*s_dot(t)**2 -r*np.sin(s(t))*s_2dot(t),-r*np.sin(s(t))*s_dot(t)**2+r*np.cos(s(t))*s_2dot(t)])


# In[3]:

T=10
steps = 3000
time = np.linspace(0,T,steps)
step_size = T/steps

m1 = 2;
m2 = 2;
l1 = 1;
l2 = 1;
d1 = 0.5;
d2 = 0.5;
I1 = 1;
I2 = 1;


# In[4]:

# Robot Model
# Physicall parameters

m1 = 2
m2 = 2
l1 = 1
l2 = 1
d1 = 0.5
d2 = 0.5
I1 = 1
I2 = 1

g0 = 9.81
#g0=0.0

# Dynamic Coefficients
a1 = I1+I2+m1*d1**2+m2*d2**2+m2*l1**2
a2 = m2*l1*d2
a3 = I2+m2*d2**2
a4 = g0*(m1*d1+m2*l1)
a5 = m2*d2*g0

def B(q): return np.array([[a1+2*a2*np.cos(q[1]),a2*np.cos(q[1])+a3],[a2*np.cos(q[1])+a3,a3]])
def c(q,q_dot): return np.array([-a2*np.sin(q[1])*(q_dot[1]**2 + 2*q_dot[0]*q_dot[1]), a2*np.sin(q[1])*q_dot[0]**2])
def g(q): return np.array([a4*np.cos(q[0])+a5*np.cos(q[0]+q[1]),a5*np.cos(q[0]+q[1])])


# In[5]:

def inv_dyn(q, q_dot, u):
    q=np.asarray(q)
    q_dot=np.asarray(q_dot)
    u=np.asarray(u)
  #  print(np.shape(c(q,q_dot)))
  #  print(np.shape(u))

    q=q.reshape(2)
    q_dot=q_dot.reshape(2)
 #   print(np.shape(u - c(q,q_dot) - g(q)))

    q_ddot= np.dot(np.linalg.inv(B(q)),u - c(q,q_dot) - g(q))
    


    q += (q_dot*step_size + ( q_ddot*step_size**2) /2)
    q_dot += step_size*q_ddot

    return q,q_dot

# I/O Dimesion and Placeholders
robot_dim = len(q_d(0))
horizon = 10
# TODO 2*
state_dim = (2 + horizon) * robot_dim + 4 
output_dim = 1


agent_state= tf.placeholder(tf.float32,shape=[None,state_dim])
Q = tf.placeholder(tf.float32,shape=[None,output_dim])

#agent_velocity = tf.placeholder(tf.float32,shape=[None,state_dim+1])
#Q_v3 = tf.placeholder(tf.float32,shape=[None,output_dim])

# def future_trajectory(time):

#     q = []
#     q_dot = []

#     for i in range(horizon):

#         q.append(q_d(time + (i+1)*step_size)) 
#         q_dot.append(q_d_dot(time + (i+1)*step_size))

#     q = np.asarray(q)
#     q_dot = np.asarray(q_dot)

#     return q.ravel(),q_dot.ravel()
def future_trajectory(time):

    
    q_ddot = []

    for i in range(horizon):

        q_ddot.append(q_2d_dot(time + (i+1)*step_size))

    q_ddot = np.asarray(q_ddot)

    return q_ddot.ravel()

## WEIGHTS

def weight_function(shape):
    
    w = tf.truncated_normal(shape)
    return tf.Variable(w)

def bias_function(shape):
    b = tf.zeros((shape))
    return tf.Variable(b)

dim_layer1 = 128

#Position Weights
w1 = weight_function([state_dim,dim_layer1])
w_out = weight_function([dim_layer1,output_dim])

b1 = bias_function([dim_layer1])

b_out = bias_function([output_dim])

def Q_network(x):

    hl1 = tf.add(tf.matmul(x,w1),b1)
    hl1 = tf.nn.tanh(hl1)

    ol = tf.add(tf.matmul(hl1,w_out),b_out)
    return ol

q_nn = Q_network(agent_state)

learning_rate = 0.1

loss = tf.reduce_mean(tf.square(Q-q_nn))

opt = tf.train.GradientDescentOptimizer(learning_rate)

#opt = tf.train.AdamOptimizer()

train = opt.minimize(loss)

# TODO tanh ?

def reward(q,q_dot,time): return np.tanh(- np.linalg.norm(q_d(time)-q) - np.linalg.norm(q_d_dot(time)-q_dot))

# Discrete Actions

actions_position= [-3,-2,-1,0,1,2,3]
actions_velocity = [-0.6,-0.4,-0.2,0,0.2,0.4,0.6]

## TRAINING

initializzer = tf.global_variables_initializer()

sess = tf.Session()

initializzer.run(session=sess)

#Expoitation Exploration

def gainsSaturation(Kp,Kd):

    if Kp < 10:

        Kp = 30

    if Kd < 2:

       Kd = 6

    if Kp > 200:

        Kp = 200

    if Kd > 40:

        Kd = 40

    return Kp,Kd

def run(state,time):

    Kp = state[0][2*robot_dim ]
    Kd = state[0][2*robot_dim +1]

    a_position = state[0][2*robot_dim+2]
    a_velocity = state[0][2*robot_dim+3]
    
    Kp += a_position
    Kd += a_velocity

    q = state[0][0:robot_dim]
    q_dot = state[0][robot_dim:2*robot_dim]

    # Initial Loss

    r = - reward(q,q_dot,time)

    t = time
    Kp,Kd = gainsSaturation(Kp, Kd)
    for idx in range(horizon):

        # Clip Time

        if ( t > T):

            t = t - T

        u = Kp*(q_d(t)-q) + Kd*(q_d_dot(t)-q_dot)
        
        r += reward(q,q_dot,t)

        q,q_dot = inv_dyn(q,q_dot,u)

        t += step_size

    state[0][0:robot_dim] = q

    state[0][robot_dim:2*robot_dim] = q_dot 

    state[0][2*robot_dim] = Kp

    state[0][2*robot_dim + 1] = Kd

    return r,state

runs = 50
Kp = 50
Kd = 10

epsilon = 0.9

for episode in range(runs):

    Kp = 50
    Kd = 10

    q = q_d(0)
    q_dot = q_d_dot(0)

    err = 0

    Kp_story = []
    Kd_story = []

    tempo = []

    for t_step in enumerate(time[0:len(time):horizon]) :

        t = t_step[1]

        q_ddot_horizon = future_trajectory(t)

        # Best action selection : Grid search

        policy_selection = np.random.rand()

        Kp,Kd = gainsSaturation(Kp, Kd)

        if policy_selection <= epsilon :

            a_p_idx = np.random.randint(0,7)
            a_v_idx = np.random.randint(0,7)

            gains=np.asarray([Kp,Kd])

            actions = np.asarray([actions_position[a_p_idx],actions_velocity[a_v_idx]])

            state_action = np.concatenate((q,q_dot,gains,actions,q_ddot_horizon))

            state_action = state_action.reshape((1,len(state_action)))


        else :
            
            Q_s_a = float(-1000000000)

            for p in range(len(actions_position)):

                for v in range(len(actions_velocity)):

                    ap = actions_position[p]
                    av = actions_velocity[v]

                    gains=np.asarray([Kp,Kd])

                    actions = np.asarray([ap,av])

                    state_t = np.concatenate((q,q_dot,gains,actions,q_ddot_horizon))

                    state_t = state_t.reshape((1,len(state_t)))

                    value = sess.run(q_nn,{agent_state:state_t})

                    if value > Q_s_a :

                        Q_s_a = value

                        state_action = state_t

        total_reward,state_t_1 = run(state_action,t)

        total_reward = total_reward.reshape((1,1))

        loss_ = sess.run(loss,{agent_state:state_action,Q : total_reward})

        sess.run(train,{agent_state:state_action,Q : total_reward})

        err += loss_

        Kp_story.append(Kp)
        Kd_story.append(Kd)

        Kp = state_t_1[0][2*robot_dim]
        Kd = state_t_1[0][2*robot_dim + 1]

        plt.scatter(q_d(t)[0],q_d(t)[1],color='blue')
        plt.scatter(q[0],q[1],color='red')
        
        q = state_t_1[0][0:robot_dim]
        q_dot = state_t_1[0][robot_dim:2*robot_dim]

        tempo.append(t_step[0])
    if episode % 3 == 0 :

        epsilon *= 0.7

    err += np.linalg.norm(q_d(T)-q) + np.linalg.norm(q_d_dot(T)-q_dot)
    plt.scatter(q_d(T)[0],q_d(T)[1],color='blue')
    plt.scatter(q[0],q[1],color='red')
    plt.show()
    plt.plot(tempo,Kp_story,color='red')
    plt.plot(tempo,Kd_story,color='blue')
    plt.show()
    print("Episode : " + str(episode +1) + "------Total Loss : " + str(err) + "------Epsilon : " + str(epsilon))

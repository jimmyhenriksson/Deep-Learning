from stock_env import Exchange
import gym
import pandas as pd 
import numpy as np

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from Models import MLP


if __name__ == "__main__":

    x = np.load('../data/processed/OMXS_processed.npy')
    y = np.load('../data/processed/OMXS_labels.npy')

    n, d = x.shape

    df = pd.DataFrame(x)
    # print(df.head())
    lookback=0
    env = Exchange(df,lookback)
    nb_actions = env.action_space.shape[0]


    # obs = env.reset()
    # for i in range(100):
    #     action = 2
    #     percentage = 0.5
    #     obs, rewards, done, info = env.step([action,percentage])
    #     env.render()

    model = Sequential()
    model.add(Dense(16, input_shape=(1,lookback+2,d)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    print(model.summary())


    memory = SequentialMemory(limit=50000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format("stock_env"), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)

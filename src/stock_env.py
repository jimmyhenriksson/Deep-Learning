import numpy as np
import pandas as pd
import gym
from gym import spaces

INITIAL_BALANCE = 20000

class Exchange(gym.Env):
    metadata = {'render.modes': ['human']}
    BUY, SELL, HOLD = 0, 1, 2


    def __init__(self, df, lookback):

        # Attributes
        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.df = df
        self.n_samples, self.n_dimension = df.shape
        self.lookback = lookback
        self.current_step = np.random.randint(self.lookback, self.n_samples)
        self.current_price = 1211.79
        self.interest_rate = 0.08/365

        # Define the action space
        # Actions[0] ([0,1,2])      : Buy, Sell, Hold
        # Actions[1] (scalar [0,1]) : Buy   - buy 100% of outstanding cash
        #                           : Sell  - sell 100% of current holdings
        #                           : Hold  - No effect
        # self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([2, 1]))
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([2]))

        # Observe the price into the lookback period
        # Plus an observation vector of current wealth, etc
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.lookback + 1, self.lookback + 1))

        super(Exchange,self).__init__()

    
    def _get_next_observations(self):
        """
        Returns the observation of the next time step
        args:
            None
        returns:
            None
        """
        prices = self.df.loc[self.current_step: self.current_step + self.lookback].values

        current_state = np.zeros((self.n_dimension))
        current_state[0] = self.balance
        current_state[1] = self.net_worth
        current_state[2] = self.shares_held

        obs = np.vstack((prices, current_state))

        return obs


    def take_action(self, action):
        """
        Updates the environment based on the action take
        args:
            action: action to take
        returns:
            None
        """

        print(f'Action taken: {action}')

        # move = action[0]
        # percentage = action[1]
        move = action
        percentage = 0.25

        if move == self.BUY:
            shares_purchasable = self.net_worth / self.current_price
            shares_purchased = int(np.floor(shares_purchasable * percentage))
            self.shares_held += shares_purchased
            self.balance -= shares_purchased * self.current_price
            pass
        elif move == self.SELL:
            shares_sold = int(np.floor(self.shares_held * percentage))
            self.shares_held -= shares_sold
            self.balance += shares_sold * self.current_price
            pass
        elif move == self.HOLD:
            # Do nothing
            pass

        self.net_worth = self.balance + (self.shares_held * self.current_price)


    def step(self, action):
        """
        Takes an action in the environment
        args:
            action (vector): the current actiont to take
        returns:
            reward: reward for taking current action
            obs: observations resulting from the action taken
            done: if the current session is complete
            debug: dictionary of debug values
        """


        # Take the action
        self.take_action(action)
        self.current_step += 1

        # If we have reached the end of this pass, reset the steps
        if self.current_step > self.n_samples:
            self.current_step = np.random.randint(self.lookback, self.n_samples)

        # Severe punishment for losing partnership capital
        if self.net_worth < 0:
            reward = -float('inf')
        else:
            reward = self.net_worth 
        print("HERE")

        # Simulation is over if agent is out of capital
        done = self.net_worth < 0
        obs = self._get_next_observations()
        debug = {}

        print(f'obs: {obs.shape}')

        return reward, obs, done, debug


    def reset(self):
        """
        Prepares the environment for initial run by resetting parameters
        args:
            None
        returns:
            None
        """
        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.shares_held = 0
        self.current_step = np.random.randint(self.lookback, self.n_samples)

        return self._get_next_observations()

    
    def render(self, mode='human'):
        """
        Displays the current human
        args
            mode: Configuration mode
                    Use Human for human consumption
        returns:
            None
        
        """
        print(f'Net worth: {self.net_worth}')
        print(f'Shares held: {self.shares_held}')
        print(f'Profits: {self.net_worth - INITIAL_BALANCE} ')
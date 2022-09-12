import numpy as np
import matplotlib.pyplot as plt
from utils import ChatBot, argument_node, generate_graph

class Environment:
    def __init__(self, structure=1):
        # Structure 1
        if structure == 1:
            self.arguments = ['a','b','c','d','e','S']
            self.relations = [['a','b'],['b','c'],['c','d'],['c','e']]
        
        # Structure 2
        elif structure == 2:
            self.arguments = ['a','b','c','d','e','f','g','S']
            self.relations = [['a','b'],['b','c'],['c','d'],['d','e'],['b','f'],['c','g']]
        
        # Structure 3
        elif structure == 3:
            self.arguments = ['a','b','c','d','e','f','g','h','S']
            self.relations = [['a','b'],['b','d'],['a','c'],['c','e'],['c','f'],['c','g'],['f','h']]
            
        # Structure 4
        elif structure == 4:
            self.arguments = ['a','b','c','d','e','f','g','h','i','j','k','l','m','S']
            self.relations = [['a','b'],['b','c'],['b','d'],['c','e'],['c','f'],['c','g'],['d','h'],['d','i'],['e','j'],['e','k'],['f','l'],['h','m']]
            
        # generated graphs
        else:
            self.arguments = structure[0]
            self.relations = structure[1]
        
    def possible_action(self, state, pool):
        action_id_list = []
        parent_dict = {}
        for arg in state:
            for relation in self.relations:
                if (arg == relation[0]) and (relation[1] in pool): # find counterargument and no repeatition
                    action_id = ord(relation[1])-97
                    action_id_list.append(action_id) # change the letter to number
                    parent_dict[action_id] = arg
        action_id_list.append(len(self.arguments)-1) # add 'stop' option
        return action_id_list, parent_dict
        
    def step(self, agent, state, pool, node_dict, leaf_dict, is_chat_bot=True, is_train=True):
        # first, agent action
        action_id_list, parent_dict = self.possible_action(state, pool)
        agent_action_id = agent.get_action(state, action_id_list, is_train=is_train)
        agent_state = state + self.arguments[agent_action_id]
        
        # if the agent chooses to stop
        if agent_action_id == len(self.arguments)-1:
            done = True
            for arg in leaf_dict:
                leaf_dict[arg].is_win = True # leaf nodes always win
            reward = self.get_reward(agent_state, agent_action_id, node_dict, is_chat_bot=is_chat_bot)
            return agent_state, reward, done, pool, agent_action_id, node_dict, leaf_dict
        
        # not chossing stop, update intermediate variables
        pool.remove(self.arguments[agent_action_id])
        last_arg = parent_dict[agent_action_id]
        last_node = node_dict[last_arg]
        new_node = argument_node(argument=self.arguments[agent_action_id], parent=last_node)
        last_node.add_child(new_node)
        node_dict[self.arguments[agent_action_id]] = new_node
        if last_arg in leaf_dict:
            del leaf_dict[last_arg]
        leaf_dict[self.arguments[agent_action_id]] = new_node
        
        # if no counter arguments or no argument in the pool, end the game
        temp_action_list, temp_parent_dict = self.possible_action(agent_state, pool)
        if len(temp_parent_dict) == 0: # no counter argument includes situation that no arguments in the pool
            done = True
            next_state = agent_state + "E"
            for arg in leaf_dict:
                leaf_dict[arg].is_win = True # leaf nodes always win
            reward = self.get_reward(next_state, agent_action_id, node_dict, is_chat_bot=is_chat_bot)
            return next_state, reward, done, pool, agent_action_id, node_dict, leaf_dict
        
        # neither agent or user chooses to stop, and pool is not empty, game continues
        done = False
        reward = 0
        return agent_state, reward, done, pool, agent_action_id, node_dict, leaf_dict
    
    def get_reward(self, state, action_id, node_dict, is_chat_bot=True):
        reward = 0
        if is_chat_bot:
            # The game ends with no counter argument or no argument in pool
            if state[-1] == 'E':
                state = state[:-1]
                # game ends after bot choosing
                if len(state)%2 == 1:
                    reward += 10
                # game ends after user choosing
                elif len(state)%2 == 0:
                    reward -= 10
                    
            if len(state) % 2 == 1: # action made by agent
                if action_id == len(self.arguments)-1: # if it is stop
                    reward -= 50
                    state = state[:-1]
                # determine whether this state is win or lose through the recursive function
                is_win = node_dict[state[0]].is_winning()
                if is_win:
                    reward += 100
                elif not is_win:
                    reward -= 100
                    
            elif len(state) % 2 == 0: # action made by user
                if action_id == len(self.arguments)-1:
                    reward += 50
                    state = state[:-1]
                is_win = node_dict[state[0]].is_winning()
                if is_win:
                    reward += 100
                elif not is_win:
                    reward -= 100
        else:
            if state[-1] == 'E':
                state = state[:-1]
                # game ends after bot choosing, reverse
                if len(state)%2 == 1:
                    reward -= 10
                # game ends after user choosing, reverse
                elif len(state)%2 == 0:
                    reward += 10
                    
            if len(state) % 2 == 1: # action made by agent
                if action_id == len(self.arguments)-1: # if it is stop
                    reward += 50
                    state = state[:-1]
                # determine whether this state is win or lose through the recursive function, to chat bot, so reverse to user
                is_win = node_dict[state[0]].is_winning()
                if is_win:
                    reward -= 100
                elif not is_win:
                    reward += 100
                    
            elif len(state) % 2 == 0: # action made by user
                if action_id == len(self.arguments)-1:
                    reward -= 50
                    state = state[:-1]
                is_win = node_dict[state[0]].is_winning()
                if is_win:
                    reward -= 100
                elif not is_win:
                    reward += 100
        return reward
    

if __name__ == "__main__":
    np.random.seed(8) # 10
    # parameter assigning
    epsilon = 0.1
    lrs = [0.05, 0.1, 0.5]
    gamma = 0.9
    structures = [5,6,7,8,9]#[1,2,3,4]
    
    for structure in structures:
        generated_arguments, generated_relations = generate_graph(np.random.choice([3,4,5,6,7]),np.random.choice([2,3]))
        generated_arguments.append('S')
        print(generated_arguments, generated_relations)
        # best_converged_winning_rate = [[],0,0,[],[],[]] # for different structures, best score should be clear. Six elements are set of parameters, average winning rate, expected winning rate, winning rate list, losing rate list, histogram data
        best_converged_winning_rate = [[],0,0,[],[]]
        for usr_lr in lrs:
            for lr in lrs:
                # initialization
                env = Environment(structure=[generated_arguments, generated_relations])
                arguments = env.arguments
                agent = ChatBot(actions=arguments, epsilon=epsilon, lr=lr, discount=gamma)
                user = ChatBot(actions=arguments, epsilon=epsilon, lr=usr_lr, discount=gamma)
                dialogue_dict = {'win':{}, 'lose':{}}
                winning_rate = []
                losing_rate = []
                state_dict = {'win':[], 'lose':[]}
                win_count = 0
                lose_count = 0
                converged_win_count = 0
                # hist_data = []
                # count = 0
                for episode in range(50000):
                    # determine the init state. a is always the key argument
                    usr_choice_pool = []
                    for relation in env.relations:
                        if relation[0] == 'a':
                            usr_choice_pool.append(arguments.index(relation[1]))
                    usr_action = arguments[user.get_action('a', usr_choice_pool)]
                    init_state = 'a' + usr_action
                    # initialize the pool and the argument tree
                    pool = arguments * 1
                    node_dict = {}
                    for arg in init_state:
                        pool.remove(arg)
                        node_dict[arg] = argument_node(argument=arg)
                    node_dict['a'].child1 = node_dict[usr_action]
                    node_dict[usr_action].parent = node_dict['a']
                    state = init_state
                    leaf_dict = {}
                    leaf_dict[usr_action] = node_dict[usr_action]
                    # update q-table
                    while True:
                        next_state, reward, done, pool, agent_action_id, node_dict, leaf_dict = env.step(agent, state, pool, node_dict, leaf_dict, is_chat_bot=True)
                        agent.learn(state, agent_action_id, reward, next_state)
                        state = next_state
                        
                        if done:
                            break
                        
                        next_state, reward, done, pool, usr_action_id, node_dict, leaf_dict = env.step(user, state, pool, node_dict, leaf_dict, is_chat_bot=False)
                        user.learn(state, usr_action_id, reward, next_state)
                        state = next_state
                        
                        if done:
                            break
                    
                    
                    # calculate winning rates and state statistics
                    # if (node_dict['a'].is_win == True) or (('E' in state) and (len(state)%2==0)) or (('S' in state) and (len(state)%2==0)):
                    #     win_count += 1
                    #     if state not in dialogue_dict['win']:
                    #         dialogue_dict['win'][state] = 1
                    #     elif state in dialogue_dict['win']:
                    #         dialogue_dict['win'][state] += 1
                    #     if episode >= 5000:
                    #         converged_win_count += 1
                    # elif (node_dict['a'].is_win == False) or (('E' in state) and (len(state)%2==1)) or (('S' in state) and (len(state)%2==1)):
                    #     lose_count += 1
                    #     if state not in dialogue_dict['lose']:
                    #         dialogue_dict['lose'][state] = 1
                    #     elif state in dialogue_dict['lose']:
                    #         dialogue_dict['lose'][state] += 1
                    
                    # hist_data.append(len(state))
                    # if len(state) == 4:
                    #     count += 1
                    #     print(count)
                    if (node_dict['a'].is_win == True):
                        win_count += 1
                        
                        state_set = set()
                        for s in state:
                            if s == 'S' or s == 'E':
                                continue
                            state_set.add(s)
                        if state_set not in state_dict['win']:
                            state_dict['win'].append(state_set)
                            
                        if state not in dialogue_dict['win']:
                            dialogue_dict['win'][state] = 1
                        elif state in dialogue_dict['win']:
                            dialogue_dict['win'][state] += 1
                        if episode >= 25000:
                            converged_win_count += 1
                    elif (node_dict['a'].is_win == False):
                        lose_count += 1
                        
                        state_set = set()
                        for s in state:
                            if s == 'S' or s == 'E':
                                continue
                            state_set.add(s)
                        if state_set not in state_dict['lose']:
                            state_dict['lose'].append(state_set)
                        
                        if state not in dialogue_dict['lose']:
                            dialogue_dict['lose'][state] = 1
                        elif state in dialogue_dict['lose']:
                            dialogue_dict['lose'][state] += 1
                        
                    if (episode == 9):
                        winning_rate.append(win_count / (episode+1))
                        losing_rate.append(lose_count / (episode+1))
                    
                    if episode % 500 == 499:
                        winning_rate.append(win_count / 500)
                        losing_rate.append(lose_count / 500)
                        win_count = 0
                        lose_count = 0
                
                # record parameters with best converged winning rate
                converged_winning_rate = converged_win_count/25000
                expected_winning_rate = len(state_dict['win'])/(len(state_dict['win'])+len(state_dict['lose']))
                if converged_winning_rate > best_converged_winning_rate[1]:
                    best_converged_winning_rate[0] = [lr,usr_lr]
                    best_converged_winning_rate[1] = converged_winning_rate
                    best_converged_winning_rate[2] = expected_winning_rate
                    best_converged_winning_rate[3] = winning_rate
                    best_converged_winning_rate[4] = losing_rate
                    # best_converged_winning_rate[5] = hist_data
                    
        # draw the best converged winning rate chart
        episodes2 = [10]
        episodes3 = [i for i in range(500,50001,500)]
        episodes = episodes2 + episodes3
        plt.plot(episodes, best_converged_winning_rate[3], label='ChatBot winning rate')
        plt.plot(episodes, best_converged_winning_rate[4], label='User winning rate')
        plt.plot(episodes, [best_converged_winning_rate[2]]*len(episodes), label='Expected winning rate')
        lr = best_converged_winning_rate[0][0]
        usr_lr = best_converged_winning_rate[0][1]
        plt.title("chat_bot_lr={0},user_lr={1}".format(lr, usr_lr))
        plt.ylabel("winning rate for every 100 episodes")
        plt.xlabel("episodes")
        plt.legend(loc=4)
        plt.grid()
        plt.savefig('/home/minyi/img/random/lr{0}_usr_lr{1}_structure{2}.png'.format(lr, usr_lr, structure))
        plt.show()
        
        print('The average winning rate is', best_converged_winning_rate[1])
        print('The expected winning rate is', best_converged_winning_rate[2])
        # print(dialogue_dict)
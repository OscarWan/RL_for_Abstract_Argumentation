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
        
    def step(self, state, pool, node_dict, leaf_dict, is_train=True):
        # first, agent action
        action_id_list, parent_dict = self.possible_action(state, pool)
        agent_action_id = action_id_list[np.random.choice(len(action_id_list))] # random policy to move
        agent_state = state + self.arguments[agent_action_id]
        
        # if the agent chooses to stop
        if agent_action_id == len(self.arguments)-1:
            done = True
            for arg in leaf_dict:
                leaf_dict[arg].is_win = True # leaf nodes always win
            return agent_state, done, pool, agent_action_id, node_dict, leaf_dict
        
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
            return next_state, done, pool, agent_action_id, node_dict, leaf_dict
        
        # if counter arguments exist in the pool, user term
        usr_action_id_list, parent_dict = self.possible_action(agent_state, pool)
        usr_action_id = usr_action_id_list[np.random.choice(len(usr_action_id_list))] # random policy to move
        usr_state = agent_state + self.arguments[usr_action_id]
        
        # user chooses to stop
        if usr_action_id == len(self.arguments)-1:
            done = True
            for arg in leaf_dict:
                leaf_dict[arg].is_win = True # leaf nodes always win
            return usr_state, done, pool, agent_action_id, node_dict, leaf_dict
        
        # not chossing stop, update intermediate variables
        pool.remove(self.arguments[usr_action_id])
        last_arg = parent_dict[usr_action_id]
        last_node = node_dict[last_arg]
        new_node = argument_node(argument=self.arguments[usr_action_id], parent=last_node)
        last_node.add_child(new_node)
        node_dict[self.arguments[usr_action_id]] = new_node
        if last_arg in leaf_dict:
            del leaf_dict[last_arg]
        leaf_dict[self.arguments[usr_action_id]] = new_node
        
        # if no counter arguments or no argument in the pool, end the game
        temp_action_list, temp_parent_dict = self.possible_action(usr_state, pool)
        if len(temp_parent_dict) == 0: # no counter argument includes situation that no arguments in the pool
            done = True
            next_state = usr_state + "E"
            for arg in leaf_dict:
                leaf_dict[arg].is_win = True # leaf nodes always win
            return next_state, done, pool, agent_action_id, node_dict, leaf_dict
        
        # neither agent or user chooses to stop, and pool is not empty, game continues
        done = False
        return usr_state, done, pool, agent_action_id, node_dict, leaf_dict


if __name__ == "__main__":
    np.random.seed(8) # 10
    # parameter assigning
    structures = [5,6,7,8,9] #[1,2,3,4]
    
    for structure in structures:
        generated_arguments, generated_relations = generate_graph(np.random.choice([3,4,5,6,7]),np.random.choice([2,3]))
        generated_arguments.append('S')
        print(generated_arguments, generated_relations)
        # best_converged_winning_rate = [[],0,0,[],[],[]] # for different structures, best score should be clear. Six elements are set of parameters, average winning rate, expected winning rate, winning rate list, losing rate list, histogram data
        # best_converged_winning_rate = [[],0,0,[],[]]
        # initialization
        env = Environment(structure=[generated_arguments, generated_relations])
        arguments = env.arguments
        # agent = ChatBot(actions=arguments, epsilon=epsilon, lr=lr, discount=gamma)
        dialogue_dict = {'win':{}, 'lose':{}}
        winning_rate = []
        losing_rate = []
        state_dict = {'win':[], 'lose':[]}
        win_count = 0
        lose_count = 0
        converged_win_count = 0
        # hist_data = []
        # count = 0
        for episode in range(10000):
            # determine the init state. a is always the key argument
            usr_choice_pool = []
            for relation in env.relations:
                if relation[0] == 'a':
                    usr_choice_pool.append(relation[1])
            usr_action = usr_choice_pool[np.random.choice(len(usr_choice_pool))] # here user takes random policy from the pool
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
                next_state, done, pool, agent_action_id, node_dict, leaf_dict = env.step(state, pool, node_dict, leaf_dict)
                state = next_state
                
                if done:
                    break
            
            if (node_dict['a'].is_winning() == True):
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
                if episode >= 5000:
                    converged_win_count += 1
            elif (node_dict['a'].is_winning() == False):
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
            
            if episode % 100 == 99:
                winning_rate.append(win_count / 100)
                losing_rate.append(lose_count / 100)
                win_count = 0
                lose_count = 0
                
        
        # record parameters with best converged winning rate
        converged_winning_rate = converged_win_count/5000
        expected_winning_rate = len(state_dict['win'])/(len(state_dict['win'])+len(state_dict['lose']))
                    
        # draw winning rate chart
        episodes = [i for i in range(100,10001,100)]
        plt.plot(episodes, winning_rate, label='ChatBot winning rate')
        plt.plot(episodes, losing_rate, label='User winning rate')
        plt.plot(episodes, [expected_winning_rate]*len(episodes), label='Expected winning rate')
        plt.title("random-random")
        plt.ylabel("winning rate for every 100 episodes")
        plt.xlabel("episodes")
        plt.legend(loc=7)
        plt.grid()
        plt.savefig('/home/minyi/img/random/random_random_structure{0}.png'.format(structure))
        plt.show()
        
        print('The average winning rate is', converged_winning_rate)
        print('The expected winning rate is', expected_winning_rate)
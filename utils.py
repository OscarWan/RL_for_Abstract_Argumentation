import numpy as np

class ChatBot:
    def __init__(self, actions, lr=0.01, discount=0.9, epsilon=0.5, q_table=None):
        self.actions = actions # argument list a~e and stop
        self.lr = lr
        self.discount_factor = discount
        self.epsilon = epsilon
        if q_table is None:
            self.q_table = {}
        else:
            self.q_table = q_table
        
    def get_action(self, state, action_id_list, is_train=True):
        if is_train:
            # if the current state (ending with user's action) is not in the q_table, add the new state
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.actions))
            # epsilon-greedy policy:
            if np.random.rand() < self.epsilon:
                action_id = action_id_list[np.random.choice(len(action_id_list))]
            else:
                # exploit, choose from q-table
                action_score_list = self.q_table[state][action_id_list]
                max_q_ind_list = np.where(action_score_list==max(action_score_list))[0]
                action_id = action_id_list[np.random.choice(max_q_ind_list)]
        else:
            # in testing, always exploit if the state exists in the q_table; if not, then random selection
            if state in self.q_table:
                action_score_list = self.q_table[state][action_id_list]
                max_q_ind_list = np.where(action_score_list==max(action_score_list))[0]
                action_id = action_id_list[np.random.choice(max_q_ind_list)]
            else:
                action_id = action_id_list[np.random.choice(len(action_id_list))]
        return action_id
    
    def learn(self, state, agent_action_id, reward, next_state):
        current_q = self.q_table[state][agent_action_id]
        if next_state not in self.q_table: # add new state to the dictionary
            self.q_table[next_state] = np.zeros(len(self.actions))
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][agent_action_id] += self.lr * (new_q - current_q)
        
class argument_node:
    def __init__(self, argument, parent=None, child1=None, child2=None, child3=None, is_win=None):
        self.argument = argument
        self.parent = parent
        self.child1 = child1
        self.child2 = child2
        self.child3 = child3
        self.is_win = is_win
    
    def add_child(self, node):
        '''
        add node to the previous one's child
        '''
        if self.child1 is None:
            self.child1 = node
        elif self.child2 is None:
            self.child2 = node
        elif self.child3 is None:
            self.child3 = node
    
    def find_children(self):
        children = []
        if self.child1:
            children.append(self.child1)
        if self.child2:
            children.append(self.child2)
        if self.child3:
            children.append(self.child3)
        return children
    
    def is_winning(self):
        # if it has is_win state, just return it
        if self.is_win is not None:
            return self.is_win
        # if not, determine for each branch
        children = self.find_children()
        is_win = False
        for child in children:
            # only all children lose, then the node wins
            is_win = is_win or child.is_winning() # any one child's win will cause the negation of this state "False", meaning the node loses
        self.is_win = not is_win
        return self.is_win
    
def generate_graph(depth=None, max_child_num=None):
    if depth is None:
        depth = np.random.randint(3,8)
    if max_child_num is None:
        max_child_num = np.random.randint(1,4)
    # numpy randint will exclude the high value
    max_child_num += 1
    
    arguments = ['a']
    relations = []
    last_depth_arg = ['a']
    while depth > 1:
        if len(arguments) >= 26:
            break
        same_depth_arg = []
        for arg_ind in range(len(last_depth_arg)):
            if len(arguments) >= 26:
                break
            child_num = np.random.randint(1, max_child_num)
            parent = last_depth_arg[arg_ind]
            for child_ind in range(child_num):
                if len(arguments) == 26:
                    break
                new_argument = chr(97+len(arguments))
                arguments.append(new_argument)
                same_depth_arg.append(new_argument)
                relations.append([parent, new_argument])
        last_depth_arg = same_depth_arg
        depth -= 1
    return arguments, relations
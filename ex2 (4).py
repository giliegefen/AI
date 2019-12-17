import copy
import math
import bisect
ids = ["203085436", "304755671"]

class Problem(object):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

class SokobanProblem(Problem):
    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation"""
        self.all_pos = []
        ma = self.to_dict(initial)
        state = (self.frozen(ma))
        self.distances = (self.position_table(state, self.all_pos))
        Problem.__init__(self, state)

    def to_dict(self, state):  # save the player position and its pos content
        map = {}
        temp = ()
        player = []  # 17,27,37
        _99 = []  # 99
        _30 = []  # 30
        _35 = []  # 35
        _25 = []  # 25
        _20 = []  # 20,27
        _10 = []  # 10,17
        _15 = []  # 15
        i = 0
        for row in state:
            j = 0
            for pos in row:
                if pos == 17 or pos == 27 or pos == 37:
                    player = (i, j, pos)
                if pos == 30 or pos == 37:
                    temp = (i, j)
                    _30.append(temp)
                if pos == 35:
                    temp = (i, j)
                    _35.append(temp)
                if pos == 25:
                    temp = (i, j)
                    _25.append(temp)
                if pos == 99:
                    temp = (i, j)
                    _99.append(temp)
                if pos == 10 or pos == 17:
                    temp = (i, j)
                    _10.append(temp)
                if pos == 15:
                    temp = (i, j)
                    _15.append(temp)
                if pos == 20 or pos == 27:
                    temp = (i, j)
                    _20.append(temp)
                temp = (i, j)
                self.all_pos.append(temp)
                j += 1
            i += 1
        map = {0: player, 99: _99, 10: _10, 20: _20, 30: _30, 15: _15, 25: _25, 35: _35, 1: (i, j)}
        return map

    def actions(self, state):  # return actions the player can executes in the given state
        return self.around_me(state)
        """Return the actions that can be executed in the given
        state. The result would typically be a tuple, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""

    def result(self, state, action):
        if len(action) == 0:  ########deadlock
            return state
        else:
            old_state = self.unfroze(state)
            new_state = self.move(old_state, action)
            new_frozen = self.frozen(new_state)
            return new_frozen

        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

    def goal_test(self, state):
        if len(state[7]) == 0:
            return True
        else:
            return False
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""

    def h(self, node):
        dict_box = self.box_dis(node.state)
        sum_distance = self.asign(dict_box)
        sum_player = self.player_goal(node.state)
        sum = self.player_box(node.state)
        dist_sum = self.dist_sum(node.state)
        return 1 * sum_distance + 3 * dist_sum + sum_player + sum

        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        ####################

    """Feel free to add your own functions"""

    def move(self, state, action):
        if action == 'U':
            x = (-1, 0)
        if action == 'R':
            x = (0, 1)
        if action == 'D':
            x = (1, 0)
        if action == 'L':
            x = (0, -1)
        new_state = self.make_the_move(state, x)
        return new_state

    def make_the_move(self, state, x):
        i = state[0][0]
        j = state[0][1]
        pos = state[0][2]
        next_step = (i + x[0], j + x[1])
        next_next_step = (i + (2 * x[0]), j + (2 * x[1]))

        old_player_p = (i, j, pos)
        _10 = state[10]  # 10 or 17
        _99 = state[99]
        _20 = state[20]  # 20 or 27
        _30 = state[30]  # 30 or 37
        _35 = state[35]  # 35
        _25 = state[25]  # 25
        _15 = state[15]  # 15

        if next_step in _10:
            state[0] = (i + x[0], j + x[1], 17)
        elif next_step in _20:
            state[0] = (i + x[0], j + x[1], 27)
        elif next_step in _30:
            after_ice = self.after_ice(state, i, j, x, False)
            if after_ice in _10:
                state[0] = (after_ice[0], after_ice[1], 17)
            elif after_ice in _20:
                state[0] = (after_ice[0], after_ice[1], 27)
            else:
                state[0] = (after_ice[0] - x[0], after_ice[1] - x[1], 37)

        elif next_step in _15:
            _15.remove(next_step)
            _10.append(next_step)
            state[0] = (i + x[0], j + x[1], 17)
            if next_next_step in _10:
                _10.remove(next_next_step)
                _15.append(next_next_step)
            elif next_next_step in _20:
                _20.remove(next_next_step)
                _25.append(next_next_step)
            else:  # 30
                after_ice = self.after_ice(state, i, j, x, True)
                if after_ice in _10:
                    _10.remove(after_ice)
                    _15.append(after_ice)
                elif after_ice in _20:
                    _20.remove(after_ice)
                    _25.append(after_ice)
                else:
                    last_ice = after_ice[0] - x[0], after_ice[1] - x[1]
                    _30.remove(last_ice)
                    _35.append(last_ice)


        elif next_step in _25:
            _25.remove(next_step)
            _20.append(next_step)
            state[0] = (i + x[0], j + x[1], 27)
            if next_next_step in _10:
                _10.remove(next_next_step)
                _15.append(next_next_step)
            elif next_next_step in _20:
                _20.remove(next_next_step)
                _25.append(next_next_step)
            else:  # 30
                after_ice = self.after_ice(state, i, j, x, True)
                if after_ice in _10:
                    _10.remove(after_ice)
                    _15.append(after_ice)
                elif after_ice in _20:
                    _20.remove(after_ice)
                    _25.append(after_ice)
                else:
                    last_ice = after_ice[0] - x[0], after_ice[1] - x[1]
                    _30.remove(last_ice)
                    _35.append(last_ice)
        else:  # 35
            after_ice = self.after_ice(state, i + x[0], j + x[1], x, False)
            _35.remove(next_step)
            _30.append(next_step)
            if after_ice in _10:
                _15.append(after_ice)
                _10.remove(after_ice)
                state[0] = (next_step[0], next_step[1], 37)
            elif after_ice in _20:
                _20.remove(after_ice)
                _25.append(after_ice)
                state[0] = (next_step[0], next_step[1], 37)
            else:
                last_ice = after_ice[0] - x[0], after_ice[1] - x[1]
                _30.remove(last_ice)
                _35.append(last_ice)
                state[0] = (next_step[0], next_step[1], 37)

        state[10] = _10
        state[20] = _20  # 20 or 27
        state[30] = _30  # 30 or 37
        state[35] = _35  # 35
        state[25] = _25  # 25
        state[15] = _15

        return state

    def after_ice(self, state, i, j, x, next_next):
        i += x[0]
        j += x[1]
        if next_next:
            i += x[0]
            j += x[1]
        (N, M) = state[1]
        while (i, j) in state[30] and i >= 0 and j >= 0 and i < N and j < M:
            i += x[0]
            j += x[1]
        temp = (i, j)
        return temp

    def around_me(self, state):  # check if the player can move around him
        move = []
        movetup = ()
        i = state[1][0]
        j = state[1][1]
        free = state[5] + state[7] + state[9]
        box = state[11] + state[13] + state[15]

        if (i - 1, j) in free or ((i - 1, j) in box and (i - 2, j) in free):
            move.append("U")
        if (i + 1, j) in free or ((i + 1, j) in box and (i + 2, j) in free):
            move.append("D")
        if (i, j + 1) in free or ((i, j + 1) in box and (i, j + 2) in free):
            move.append("R")

        if (i, j - 1) in free or ((i, j - 1) in box and (i, j - 2) in free):
            move.append("L")
        movetup = tuple(move)
        return movetup

    def unfroze(self, state):
        dict_state = {}
        rowl = []
        for i in range(1, 18, 2):
            rowl = list(state[i])
            dict_state[state[i - 1]] = rowl
        return dict_state

    def frozen(self, state):
        tup = ()
        for k, v in state.items():
            state[k] = tuple(v)
        for k in state.items():
            tup = tup + k
        return tup

    def box_dis(self, state):
        box = state[11] + state[13] + state[15]
        goal = state[7] + state[13]
        dict_box = {}
        for b in box:
            b_dis = []
            for g in goal:
                x = abs(b[0] - g[0])
                y = abs(b[1] - g[1])
                dis = x + y
                temp = (g, dis)
                b_dis.append(temp)
            dict_box[b] = b_dis
        return dict_box

    def asign(self, dict_box):
        sum_distance = 0
        used = []
        for b in dict_box.items():
            goal = b[1]
            min = math.inf
            for dis in goal:
                if dis[1] < min and dis[0] not in used:
                    min = dis[1]
                    temp = dis[0]
            used.append(temp)
            sum_distance = sum_distance + min
        return sum_distance

    def player_goal(self, state):
        min = math.inf
        player = state[1]
        goal = state[7] + state[13]
        for g in goal:
            x = abs(player[0] - g[0])
            y = abs(player[1] - g[1])
            dis = math.sqrt(x ** 2 + y ** 2)
            if dis < min:
                min = dis
        return min

    def player_box(self, state):
        sum = 0
        min = math.inf
        player = state[1]
        box = state[11] + state[13] + state[15]
        for b in box:
            x = abs(player[0] - b[0])
            y = abs(player[1] - b[1])
            dis = x + y
            if dis < min:
                min = dis
        return min

    def init_dis(self, state, all_pos, ):
        goal = state[7] + state[13]
        goal_len = len(goal)
        pos_len = len(all_pos)
        distance_to_goal = [[math.inf for x in range(pos_len)] for y in range(goal_len)]
        i = 0
        for g in goal:
            j = 0
            for pos in all_pos:
                if g == pos:
                    distance_to_goal[i][j] = 0
                j += 1
            i += 1
        return distance_to_goal

    def position_table(self, state, all_pos):
        distance_to_goal = self.init_dis(state, all_pos)
        notwall = state[5] + state[7] + state[9] + state[11] + state[13] + state[15]
        position = []
        direct = [(-1, 0), (1, 0), (0, -1), (0, 1), ]
        goal = state[7] + state[13]
        for g in goal:
            position.append(g)
            while len(position) != 0:
                pos = position.pop(0)
                for d in direct:
                    box_pos = (pos[0] + d[0], pos[1] + d[1])
                    player_pos = ((pos[0] + (2 * d[0]), pos[1] + (2 * d[1])))
                    if box_pos in notwall and player_pos in notwall:
                        i = goal.index(g)
                        j = all_pos.index(pos)
                        k = all_pos.index(box_pos)
                        if distance_to_goal[i][k] == math.inf:
                            distance_to_goal[i][k] = distance_to_goal[i][j] + 1
                            position.append(box_pos)
        return distance_to_goal

    def dist_sum(self, state):
        used = []
        sum = 0
        temp = 0
        box = state[11] + state[13] + state[15]
        goal = state[7] + state[13]
        for b in box:
            j = self.all_pos.index(b)
            min = math.inf
            for g in goal:
                if g not in used:
                    i = goal.index(g)
                    dis = self.distances[i][j]
                    if dis < min:
                        min = dis
                        temp = g
            if temp != 0:
                used.append(temp)
            sum += min
            if sum == math.inf:
                sum = self.dist_sumTemp(state)
        return sum

    def dist_sumTemp(self, state):
        used = []
        sum = 0
        temp = 0
        box = state[11] + state[13] + state[15]
        goal = state[7] + state[13]
        for b in box:
            j = self.all_pos.index(b)
            min = math.inf
            for g in goal:
                if g not in used:
                    temp = 0
                    i = goal.index(g)
                    dis = self.distances[i][j]
                    if dis < min:
                        min = dis
                        temp = g
            if temp != 0:
                used.append(temp)
            sum += min
        return sum


class Queue:

    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)

class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

class PriorityQueue(Queue):

    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)

class SokobanController:
    def __init__(self, input_map):
        self.mid_i=0
        self.mid_j=0
        self.sum=4
        self._10 = set()
        self._20 = set()
        self.old_box = set()
        self.new_box = set()
        self.middle20=set()
        self.change_box = False
        self.dicovery = set()
        self.beenthere = set()
        self.corner = set()
        self.middle = set()
        self.allpos=self.all_pos(input_map)
        self.old_pos=(-1,-1)
        self.new_pos=(-1,-1)

        self.last_move='N'
        self.my_map=copy.deepcopy(input_map)
        self.fin=[]
        self.counter=-1
        self.moves_rec=[]
        self.orign=set()
        self.og=(-1,-1)


    def get_next_action(self, observed_map):
        mymp=[]
        if self.old_pos == (-1, -1):
            self.old_pos = self.new_p(observed_map)
            self.og=self.old_pos
            self.orign.add(self.og)
            self.new_pos=self.old_pos
        else:
            self.new_pos=self.new_p(observed_map)

        if self.last_move != 'No' and self.last_move!='result' and self.last_move!='continue':
            self.change_map(observed_map)
            self.new_box = self.new_box_p(observed_map)
            if self.old_box!=self.new_box:
                self.change_map_with_box(observed_map)
                self.old_box = self.new_box
            self.fix_map(observed_map)
            self.old_pos = self.new_pos
            move = self.where_to()
            self.last_move = move
            if self.last_move == 'No':
                mymp.append(self.totup(self.my_map))
                g = self.solve_problems(mymp)
                self.fin = g
                mm = self.fin.pop(0)
                if (len(self.middle) * 2) < self.sum and self.last_move != 'result':
                    self.last_move='result'
                else:
                    self.last_move = mm
                    return mm
            if move=='result' or self.last_move=='result':
                self.last_move='continue'
                mymp.append(self.totup(self.my_map))
                g = self.solve_problems(mymp)
                self.fin = g
                mm = self.fin.pop(0)
                self.counter += 1
                return mm

            self.counter += 1
            return move
        if self.last_move=='No':
            mymp.append(self.totup(self.my_map))
            g = self.solve_problems(mymp)
            self.fin = g
            mm = self.fin.pop(0)
            self.last_move=mm
            self.counter += 1
            return mm

        if self.last_move=='continue':
            while len(self.fin) != 0:
                self.counter += 1
                self.counter += 1
                return self.fin.pop(0)

    def fix_map(self,ob):
        i=0
        while i<(self.mid_i*2):
            j = 0
            while j<(self.mid_j*2):
                if self.my_map[i][j]!=ob[i][j]:
                    if self.my_map[i][j]<30:
                        self.my_map[i][j] = ob[i][j]
                    else:
                        self.my_map[i][j] = ob[i][j]+20

                j+=1
            i+=1
        self.og=self.new_pos

    def all_pos(self,map):
        listall=[]
        i = 0
        for row in map:
            j = 0
            for pos in row:
                temp = (i, j)
                listall.append(temp)
                j += 1
            i += 1
        ii = 0
        self.mid_i=i/2
        self.mid_j=j/2
        j=j-1
        i=i-1
        for r in map:
            jj = 0
            for k in r:
                temp = (ii, jj)
                if ii==0 or jj==0 or ii==i or jj==j:
                    self.corner.add(temp)
                else:
                    if k==10 or k==17 or k==20 or k==27:
                        self.dicovery.add(temp)
                        if k==10:
                            self.middle.add(temp)
                        if k==20:
                            self.middle20.add(temp)
                if k==10 or k==17:
                    self._10.add(temp)
                if k==20 or k==27:
                    self._20.add(temp)
                if k==15 or k==25:
                    self.old_box.add(temp)
                jj+=1
            ii+=1
        if i>2 and j>2:
            self.sum=(i-2)*(j-2)
        return listall

    def totup(self,fixed):
        tup=()
        for i in fixed:
            tup+=tuple(i),
        return tup

    def new_box_p(self,map):
        self.new_box=set()
        i = 0
        for row in map:
            j = 0
            for pos in row:
                if pos==15 or pos==25:
                    temp = (i, j)
                    self.new_box.add(temp)
                j += 1
            i += 1
        return self.new_box


    def new_p(self,ob_map):
        i = 0
        for o in ob_map:
            if 17 in o:
                j = o.index(17)
                temp = (i, j)
                return temp
                break
            if 27 in o:
                j = o.index(27)
                temp = (i, j)
                return temp
                break
            i += 1

    def where_to(self):
        mymp=[]
        i = self.old_pos[0]
        j = self.old_pos[1]
        current = (i, j)
        self.beenthere.add(current)
        U = (i - 1, j)
        D = (i + 1, j)
        R = (i, j + 1)
        L = (i, j - 1)
        flag=False
        if len(self.middle)==0:
            return 'result'
        elif (len(self.middle) * 2) < self.sum:
            return 'No'
        elif U in self.middle:
            self.beenthere.add(U)
            self.moves_rec.append('U')
            if U in self.middle:
                self.middle.remove(U)
            return 'U'
        elif L in self.middle:
            self.beenthere.add(L)
            self.moves_rec.append('L')
            if L in self.middle:
                self.middle.remove(L)
            return 'L'
        elif D in self.middle:
            self.beenthere.add(D)
            self.moves_rec.append('D')
            if D in self.middle:
                self.middle.remove(D)
            return 'D'
        elif R in self.middle:
            self.beenthere.add(R)
            self.moves_rec.append('R')
            if R in self.middle:
                self.middle.remove(R)
            return 'R'
        elif U in self.orign and U not in self.corner:
            self.beenthere.add(U)
            self.moves_rec = []
            return 'U'
        elif L in self.orign and L not in self.corner:
            self.beenthere.add(L)
            self.moves_rec = []
            return 'L'
        elif D in self.orign and D not in self.corner:
            self.beenthere.add(D)
            self.moves_rec = []
            return 'D'
        elif R in self.orign and R not in self.corner:
            self.beenthere.add(R)
            self.moves_rec = []
            return 'R'

        else:
            if i<j<self.mid_j:
                if L in self._10 and L not in self.beenthere:
                    self.beenthere.add(L)
                    self.moves_rec.append('L')
                    return 'L'
                elif D in self._10 and D not in self.beenthere:
                    self.beenthere.add(D)
                    self.moves_rec.append('D')
                    return 'D'
                elif R in self._10 and R not in self.beenthere:
                    self.beenthere.add(R)
                    self.moves_rec.append('R')
                    return 'R'

                elif U in self._10 and U not in self.beenthere:
                    self.beenthere.add(U)
                    self.moves_rec.append('U')
                    return 'U'
                else:
                    flag=True
            elif i<j:
                if U in self._10 and U not in self.beenthere:
                    self.beenthere.add(U)
                    self.moves_rec.append('U')
                    return 'U'
                elif D in self._10 and D not in self.beenthere:
                    self.beenthere.add(D)
                    self.moves_rec.append('D')
                    return 'D'
                elif L in self._10 and L not in self.beenthere:
                    self.beenthere.add(L)
                    self.moves_rec.append('L')
                    return 'L'
                elif R in self._10 and R not in self.beenthere:
                    self.beenthere.add(R)
                    self.moves_rec.append('R')
                    return 'R'

                else:
                    flag=True
            elif j<i<self.mid_j:
                if R in self._10 and R not in self.beenthere:
                    self.beenthere.add(R)
                    self.moves_rec.append('R')
                    return 'R'
                elif U in self._10 and U not in self.beenthere:
                    self.beenthere.add(U)
                    self.moves_rec.append('U')
                    return 'U'
                if D in self._10 and D not in self.beenthere:
                    self.beenthere.add(D)
                    self.moves_rec.append('D')
                    return 'D'
                elif L in self._10 and L not in self.beenthere:
                    self.beenthere.add(L)
                    self.moves_rec.append('L')
                    return 'L'
                else:
                    flag=True
            elif j<=i:
                if L in self._10  and L not in self.beenthere:
                    self.beenthere.add(L)
                    self.moves_rec.append('L')
                    return 'L'
                elif R in self._10 and R not in self.beenthere:
                    self.beenthere.add(R)
                    self.moves_rec.append('R')
                    return 'R'

                elif U in self._10 and U not in self.beenthere:
                    self.beenthere.add(U)
                    self.moves_rec.append('U')
                    return 'U'
                elif D in self._10 and D not in self.beenthere:
                    self.beenthere.add(D)
                    self.moves_rec.append('D')
                    return 'D'

                else:
                    if U in self._20 and U not in self.beenthere:
                        self.beenthere.add(U)
                        self.moves_rec.append('U')
                        return 'U'
                    elif L in self._20 and L not in self.beenthere:
                        self.beenthere.add(L)
                        self.moves_rec.append('L')
                        return 'L'
                    elif R in self._20 and R not in self.beenthere:
                        self.beenthere.add(R)
                        self.moves_rec.append('R')
                        return 'R'
                    elif D in self._20 and D not in self.beenthere:
                        self.beenthere.add(D)
                        self.moves_rec.append('D')
                        return 'D'
                    elif U in self.orign:
                        self.beenthere.add(U)
                        self.moves_rec=[]
                        return 'U'
                    elif L in self.orign:
                        self.beenthere.add(L)
                        self.moves_rec=[]
                        return 'L'
                    elif R in self.orign:
                        self.beenthere.add(R)
                        self.moves_rec=[]
                        return 'R'
                    elif D in  self.orign:
                        self.beenthere.add(D)
                        self.moves_rec=[]
                        return 'D'
                    else:
                        flag=True

            elif len(self.moves_rec) != 0:
                last = self.moves_rec.pop()
                if last == 'U':
                    self.beenthere.add(U)
                    return 'D'
                elif last == 'D':
                    self.beenthere.add(D)
                    return 'U'
                elif last == 'R':
                    self.beenthere.add(R)
                    return 'L'
                elif last == 'L':
                    self.beenthere.add(L)
                    return 'R'
                else:

                    flag=True
            else:
                flag=True

        if flag==True:
            if len(self.moves_rec) != 0:
                last = self.moves_rec.pop()
                if last == 'U':
                    self.beenthere.add(U)
                    return 'D'
                elif last == 'D':
                    self.beenthere.add(D)
                    return 'U'
                elif last == 'R':
                    self.beenthere.add(R)
                    return 'L'
                elif last == 'L':
                    self.beenthere.add(L)
                    return 'R'
                else:
                    return "No"
            else:
                return "No"


    def change_map_with_box(self, ob):
        if self.old_box!=self.new_box:
            new_place=list(self.new_box-self.old_box)
            last_place=list(self.old_box-self.new_box)
            if self.last_move == 'U':
                x = (-1, 0)
                y = abs(new_place[0][0] - last_place[0][0])
            elif self.last_move == 'D':
                x = (1, 0)
                y = abs(new_place[0][0] - last_place[0][0])
            elif self.last_move == 'R':
                x = (0, 1)
                y = abs(new_place[0][1] - last_place[0][1])
            else:
                x = (0, -1)
                y = abs(new_place[0][1] - last_place[0][1])
            if y > 1:
                i = last_place[0][0]
                j = last_place[0][1]
                count = 1
                for yy in range(y - 1):
                    self.my_map[i + count * x[0]][j + count * x[1]] = 30
                    temp = (i + count * x[0], j + count * x[1])
                    self.beenthere.add(temp)
                    if temp in self.middle:
                        self.middle.remove(temp)
                    count += 1
            if new_place[0] in self.middle:
                self.middle.remove(new_place[0])
        return self.my_map


    def change_map(self,ob):
        if self.old_pos!=(-1,-1) and self.new_pos!=(-1,-1):
            if self.last_move == 'U':
                x = (-1, 0)
                y = abs(self.new_pos[0] - self.old_pos[0])
            elif self.last_move == 'D':
                x = (1,0 )
                y = abs(self.new_pos[0] - self.old_pos[0])
            elif self.last_move == 'R':
                x = (0, 1)
                y = abs(self.new_pos[1] - self.old_pos[1])
            else:
                x = (0, -1)
                y = abs(self.new_pos[1] - self.old_pos[1])
            if y>1:
                i = self.old_pos[0]
                j = self.old_pos[1]
                count = 1
                for yy in range(y-1):
                    self.my_map[i+count*x[0]][j+count*x[1]]=30
                    temp=(i+count*x[0],j+count*x[1])
                    self.beenthere.add(temp)
                    if temp in self.middle:
                        self.middle.remove(temp)
                    count+=1
        return self.my_map
        # Should output one of the following: "U", "D", "L", "R"
        # Timeout: 5 seconds
##################################################################################
##################################################################################
##################################################################################
##################################################################################
    def create_sokoban_problem(self,game):
        return SokobanProblem(game)

    def solve_problems(self,problems):
        solved = 0
        for problem in problems:
            try:
                p = self.create_sokoban_problem(problem)
            except Exception as e:
                print("Error creating problem: ", e)
                return None
            ff=self.mid_i*2+self.mid_j*2+5
            if (ff)<16:
                result = self.check_problem(p, (lambda p: self.astar_search(p, p.h)))
            else:
                result = self.check_problem(p, (lambda p: self.best_first_graph_search(p, p.h)))

            return result

    def check_problem(self, p, search_method):
        s = self.run(search_method, args=[p])
        if isinstance(s, Node):
            solve = s
            solution = list(map(lambda n: n.action, solve.path()))[1:]
            return solution
        elif s is None:
            return (-2, -2, None)
        else:
            return s

    def run(self, func, args=(), kwargs={}):
        # remove try if you want program to abort at error
        result = (-3, -3)
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = (-3, -3, e)
        return result

    def astar_search(self,problem, h=None):
        """A* search is best-first graph search with f(n) = g(n)+h(n).
        You need to specify the h function when you call astar_search, or
        else in your Problem subclass."""
        h = self.memoize(h or problem.h, 'h')
        return self.best_first_graph_search(problem, lambda n: n.path_cost + h(n))

    def memoize(self,fn, slot=None, maxsize=32):
        """Memoize fn: make it remember the computed value for any argument list.
        If slot is specified, store result in that slot of first argument.
        If slot is false, use lru_cache for caching the values."""
        if slot:
            def memoized_fn(obj, *args):
                if hasattr(obj, slot):
                    return getattr(obj, slot)
                else:
                    val = fn(obj, *args)
                    setattr(obj, slot, val)
                    return val
        else:
            @functools.lru_cache(maxsize=maxsize)
            def memoized_fn(*args):
                return fn(*args)

        return memoized_fn

    def best_first_graph_search(self,problem, f):
        """Search the nodes with the lowest f scores first.
        You specify the function f(node) that you want to minimize; for example,
        if f is a heuristic estimate to the goal, then we have greedy best
        first search; if f is node.depth then we have breadth-first search.
        There is a subtlety: the line "f = memoize(f, 'f')" means that the f
        values will be cached on the nodes as they are computed. So after doing
        a best first search you can examine the f values of the path returned."""
        f = self.memoize(f, 'f')
        node = Node(problem.initial)
        if problem.goal_test(node.state):
            return node
        frontier = PriorityQueue(min, f)
        frontier.append(node)
        explored = set()
        while frontier:
            node = frontier.pop()
            if problem.goal_test(node.state):
                return node
            explored.add(node.state)
            for child in node.expand(problem):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                elif child in frontier:
                    incumbent = frontier[child]
                    if f(child) < f(incumbent):
                        del frontier[incumbent]
                        frontier.append(child)
        return None


###########################################
###########################################
###########################################




import search
import random
import math


ids = ["203085436", "304755671"]

class SokobanProblem(search.Problem):
    """This class implements a sokoban problem"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation"""
        self.all_pos = []
        ma=self.to_dict(initial)
        state=(self.frozen(ma))
        self.distances=(self.position_table(state,self.all_pos))
        search.Problem.__init__(self, state)

    def to_dict(self, state): #save the player position and its pos content
        map = {}
        temp=()
        player = [] #17,27,37
        _99 = []  # 99
        _30 = [] #30
        _35 = [] #35
        _25 = [] #25
        _20 = [] #20,27
        _10 = [] #10,17
        _15 =[] #15
        i=0
        for row in state:
            j = 0
            for pos in row:
                if pos == 17 or pos == 27 or pos == 37:
                    player=(i,j,pos)
                if pos==30 or pos==37:
                    temp = (i, j)
                    _30.append(temp)
                if pos==35:
                    temp = (i, j)
                    _35.append(temp)
                if pos==25:
                    temp = (i, j)
                    _25.append(temp)
                if pos==99:
                    temp = (i, j)
                    _99.append(temp)
                if pos==10 or pos==17:
                    temp = (i, j)
                    _10.append(temp)
                if pos==15:
                    temp = (i, j)
                    _15.append(temp)
                if pos==20 or pos==27:
                    temp = (i, j)
                    _20.append(temp)
                temp=(i,j)
                self.all_pos.append(temp)
                j+=1
            i+=1
        map={0: player,99:_99, 10: _10, 20:_20, 30: _30, 15: _15, 25:_25,35:_35,1:(i,j) }
        return map


    def actions(self, state): #return actions the player can executes in the given state
        return self.around_me(state)
        """Return the actions that can be executed in the given
        state. The result would typically be a tuple, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""


    def result(self, state, action):
        if len(action)==0: ########deadlock
            return state
        else:
            old_state = self.unfroze(state)
            new_state = self.move(old_state,action)
            new_frozen= self.frozen(new_state)
            return new_frozen

        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""


    def goal_test(self, state):
        if len(state[7])==0:
            return True
        else:
            return False
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""

    def h(self, node):
        dict_box=self.box_dis(node.state)
        sum_distance= self.asign(dict_box)
        sum_player=self.player_goal(node.state)
        sum=self.player_box(node.state)
        dist_sum=self.dist_sum(node.state)
        return 1*sum_distance+3*dist_sum+sum_player+sum

        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
          ####################

    """Feel free to add your own functions"""

    def move (self,state,action):
        if action=='U':
            x = (-1,0)
        if action=='R':
            x = (0,1)
        if action=='D':
            x = (1,0)
        if action=='L':
            x = (0,-1)
        new_state = self.make_the_move(state, x)
        return new_state

    def make_the_move(self, state, x):
        i = state[0][0]
        j = state[0][1]
        pos = state[0][2]
        next_step = (i + x[0], j + x[1])
        next_next_step = (i + (2 * x[0]), j + (2 * x[1]))

        old_player_p = (i, j,pos)
        _10 = state[10] #10 or 17
        _99 = state[99]
        _20 = state[20] # 20 or 27
        _30 = state[30]  #30 or 37
        _35 = state[35]  # 35
        _25 = state[25]  # 25
        _15 = state[15]  # 15

        if next_step in _10:
            state[0] = (i + x[0], j + x[1], 17)
        elif next_step in _20:
            state[0] = (i + x[0], j + x[1], 27)
        elif next_step in _30:
            after_ice = self.after_ice(state, i, j, x,False)
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
                after_ice = self.after_ice(state, i, j, x,True)
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
                after_ice = self.after_ice(state, i, j, x,True)
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
            after_ice = self.after_ice(state, i + x[0], j + x[1], x,False)
            _35.remove(next_step)
            _30.append(next_step)
            if after_ice in _10:
                _15.append(after_ice)
                _10.remove(after_ice)
                state[0] = (next_step[0] , next_step[1] , 37)
            elif after_ice in _20:
                _20.remove(after_ice)
                _25.append(after_ice)
                state[0] = (next_step[0], next_step[1], 37)
            else:
                last_ice = after_ice[0] - x[0], after_ice[1] - x[1]
                _30.remove(last_ice)
                _35.append(last_ice)
                state[0] = (next_step[0], next_step[1] , 37)


        state[10]=_10
        state[20]=_20  # 20 or 27
        state[30]=_30  # 30 or 37
        state[35]=_35  # 35
        state[25]=_25  # 25
        state[15]=_15

        return state


    def after_ice(self,state,i,j,x,next_next):
        i += x[0]
        j += x[1]
        if next_next:
            i += x[0]
            j += x[1]
        (N,M)=state[1]
        while (i,j) in state[30] and i>=0 and j>=0 and i<N and j<M:
            i += x[0]
            j += x[1]
        temp=(i,j)
        return temp



    def around_me(self, state): #check if the player can move around him
        move = []
        movetup = ()
        i = state[1][0]
        j = state[1][1]
        free= state[5] + state[7] + state[9]
        box = state[11] + state[13]+ state[15]

        if (i-1,j) in free or ((i-1,j) in box and (i-2,j) in free):
            move.append("U")
        if (i+1,j) in free or ((i+1,j) in box and (i+2,j) in free):
            move.append("D")
        if (i, j + 1) in free or ((i, j + 1) in box and (i, j + 2) in free):
            move.append("R")

        if (i,j-1) in free or ((i,j-1) in box and (i,j-2) in free):
            move.append("L")
        movetup = tuple(move)
        return movetup

    def unfroze(self,state):
        dict_state = {}
        rowl=[]
        for i in range(1,18,2):
            rowl = list(state[i])
            dict_state[state[i-1]]=rowl
        return dict_state


    def frozen(self,state):
        tup=()
        for k,v in state.items():
            state[k]=tuple(v)
        for k in state.items():
            tup=tup+k
        return tup

    def box_dis(self,state):
        box = state[11] + state[13] + state[15]
        goal = state[7]+state[13]
        dict_box={}
        for b in box:
            b_dis = []
            for g in goal:
                x=abs(b[0]-g[0])
                y=abs(b[1]-g[1])
                dis=x+y
                temp=(g,dis)
                b_dis.append(temp)
            dict_box[b]=b_dis
        return dict_box

    def asign(self,dict_box):
        sum_distance=0
        used=[]
        for b in dict_box.items():
            goal=b[1]
            min=math.inf
            for dis in goal:
                if dis[1]<min and dis[0] not in used:
                    min=dis[1]
                    temp=dis[0]
            used.append(temp)
            sum_distance=sum_distance+min
        return sum_distance

    def player_goal(self,state):
        min=math.inf
        player=state[1]
        goal=state[7]+state[13]
        for g in goal:
            x = abs(player[0] - g[0])
            y = abs(player[1] - g[1])
            dis = math.sqrt(x**2+y**2)
            if dis<min:
                min=dis
        return min

    def player_box(self,state):
        sum=0
        min=math.inf
        player = state[1]
        box = state[11] + state[13] + state[15]
        for b in box:
            x = abs(player[0] - b[0])
            y = abs(player[1] - b[1])
            dis = x+y
            if dis<min:
                min=dis
        return min

    def init_dis(self,state,all_pos,):
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

    def position_table(self,state,all_pos):
        distance_to_goal=self.init_dis(state,all_pos)
        notwall=state[5]+state[7]+state[9]+state[11]+state[13]+state[15]
        position=[]
        direct = [(-1, 0), (1, 0), (0, -1), (0, 1), ]
        goal = state[7] + state[13]
        for g in goal:
            position.append(g)
            while len(position)!=0:
                pos=position.pop(0)
                for d in direct:
                    box_pos = (pos[0] + d[0], pos[1] + d[1])
                    player_pos = ((pos[0] + (2 * d[0]), pos[1] + (2 * d[1])))
                    if box_pos in notwall and player_pos in notwall:
                        i = goal.index(g)
                        j = all_pos.index(pos)
                        k = all_pos.index(box_pos)
                        if distance_to_goal[i][k]==math.inf:
                            distance_to_goal[i][k]=distance_to_goal[i][j]+1
                            position.append(box_pos)
        return distance_to_goal

    def dist_sum(self, state):
        used=[]
        sum=0
        temp = 0
        box = state[11] + state[13] + state[15]
        goal = state[7] + state[13]
        for b in box:
            j=self.all_pos.index(b)
            min=math.inf
            for g in goal:
                if g not in used:
                    i=goal.index(g)
                    dis=self.distances[i][j]
                    if dis<min:
                        min=dis
                        temp=g
            if temp!=0:
                used.append(temp)
            sum+=min
            if sum==math.inf:
                sum=self.dist_sumTemp(state)
        return sum

    def dist_sumTemp(self, state):
        used=[]
        sum=0
        temp = 0
        box = state[11] + state[13] + state[15]
        goal = state[7] + state[13]
        for b in box:
            j=self.all_pos.index(b)
            min=math.inf
            for g in goal:
                if g not in used:
                    temp = 0
                    i=goal.index(g)
                    dis=self.distances[i][j]
                    if dis<min:
                        min=dis
                        temp=g
            if temp!=0:
                used.append(temp)
            sum+=min
        return sum


def create_sokoban_problem(game):
    return SokobanProblem(game)
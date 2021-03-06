R = -0.1
nS = 11 # 0 ~ 11
nA = 4 # 0:left 1:up 2:right 3:down
p = 0.8
t = (1-p)/2
nS = 11 # 0 ~ 11
nA = 4 # 0:left 1:up 2:right 3:down
gama = 1
theta = 0.0001
ENV = { # state:{action:{prob,next_state,reward,terminated}}
    0:{
        0:[(p,0,R,0),(t,0,R,0),(t,4,R,0)],
        1:[(p,4,R,0),(t,0,R,0),(t,1,R,0)],
        2:[(p,1,R,0),(t,4,R,0),(t,0,R,0)],
        3:[(p,0,R,0),(t,0,R,0),(t,1,R,0)]
       },
    1:{
        0:[(p,0,R,0),(t,1,R,0),(t,1,R,0)],
        1:[(p,1,R,0),(t,0,R,0),(t,2,R,0)],
        2:[(p,2,R,0),(t,1,R,0),(t,1,R,0)],
        3:[(p,1,R,0),(t,0,R,0),(t,2,R,0)]
       },
    2:{
        0:[(p,1,R,0),(t,2,R,0),(t,5,R,0)],
        1:[(p,5,R,0),(t,1,R,0),(t,3,R,0)],
        2:[(p,3,R,0),(t,5,R,0),(t,2,R,0)],
        3:[(p,2,R,0),(t,1,R,0),(t,3,R,0)]
       },
    3:{
        0:[(p,2,R,0),(t,6,-1,1),(t,3,R,0)],
        1:[(p,6,-1,1),(t,2,R,0),(t,3,R,0)],
        2:[(p,3,R,0),(t,6,-1,1),(t,3,R,0)],
        3:[(p,3,R,0),(t,2,R,0),(t,3,R,0)]
       },
    4:{
        0:[(p,4,R,0),(t,0,R,0),(t,7,R,0)],
        1:[(p,7,R,0),(t,4,R,0),(t,4,R,0)],
        2:[(p,4,R,0),(t,7,R,0),(t,0,R,0)],
        3:[(p,0,R,0),(t,4,R,0),(t,4,R,0)]
       },
    5:{
        0:[(p,5,R,0),(t,2,R,0),(t,9,R,0)],
        1:[(p,9,R,0),(t,5,R,0),(t,6,-1,1)],
        2:[(p,6,-1,1),(t,9,R,0),(t,2,R,0)],
        3:[(p,2,R,0),(t,5,R,0),(t,6,-1,1)]
       },
    6:{
        0:[(1,6,0,1)],
        1:[(1,6,0,1)],
        2:[(1,6,0,1)],
        3:[(1,6,0,1)]
       },
    7:{
        0:[(p,7,R,0),(t,7,R,0),(t,4,R,0)],
        1:[(p,7,R,0),(t,7,R,0),(t,8,R,0)],
        2:[(p,8,R,0),(t,7,R,0),(t,4,R,0)],
        3:[(p,4,R,0),(t,7,R,0),(t,8,R,0)]
       },
    8:{
        0:[(p,7,R,0),(t,8,R,0),(t,8,R,0)],
        1:[(p,8,R,0),(t,7,R,0),(t,9,R,0)],
        2:[(p,9,R,0),(t,8,R,0),(t,8,R,0)],
        3:[(p,8,R,0),(t,7,R,0),(t,9,R,0)]
       },
    9:{
        0:[(p,8,R,0),(t,5,R,0),(t,9,R,0)],
        1:[(p,9,R,0),(t,8,R,0),(t,10,1,1)],
        2:[(p,10,1,1),(t,9,R,0),(t,5,R,0)],
        3:[(p,5,R,0),(t,8,R,0),(t,10,1,1)]
       },
    10:{
        0:[(1,10,0,1)],
        1:[(1,10,0,1)],
        2:[(1,10,0,1)],
        3:[(1,10,0,1)]
       },
}
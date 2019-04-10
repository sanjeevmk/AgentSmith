def identity(state,reward):
    return reward

def mountainCar(state,reward):
    reward = state[0]+0.5
    if state[0] >= 0.5:
        reward+=1
    return reward
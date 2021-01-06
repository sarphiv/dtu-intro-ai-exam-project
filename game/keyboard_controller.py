import pygame as pg

wasd_control_scheme = [
    pg.K_w,     #Forward
    pg.K_s,     #Backward
    pg.K_a,     #Left
    pg.K_d,     #Right
]

uhjk_control_scheme = [
    pg.K_u,     #Forward
    pg.K_j,     #Backward
    pg.K_h,     #Left
    pg.K_k,     #Right
]

action_space = {
    (False, False, False, False): 0,
    (False, False, False,  True): 1,
    (False, False,  True, False): 2,
    (False,  True, False, False): 3,
    (False,  True, False,  True): 4,
    (False,  True,  True, False): 5,
    ( True, False, False, False): 6,
    ( True, False, False,  True): 7,
    ( True, False,  True, False): 8,
    
    
    (False, False,  True,  True): 0,
    ( True,  True, False, False): 0,
    
    ( True, False,  True,  True): 6,
    (False,  True,  True,  True): 3,
    
    ( True, True,   True,  False): 2,
    ( True, True,   False,  True): 1,
}


def create_keyboard_controller(events_retriever, control_scheme):
    key_state = {k: False for k in control_scheme }

    def controller():
        for e in events_retriever():
            if e.type == pg.KEYDOWN:
                if e.key in key_state:
                    key_state[e.key] = True
            elif e.type == pg.KEYUP:
                if e.key in key_state:
                    key_state[e.key] = False

        return action_space[tuple(key_state.values())]

    return controller
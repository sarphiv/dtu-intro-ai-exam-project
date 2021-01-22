import pygame as pg
from game.action_space import action_to_id

wasd_control_scheme = [
    pg.K_w,     #Forward
    pg.K_s,     #Backward
    pg.K_a,     #Left
    pg.K_d,     #Right
]

dvorak_wasd_control_scheme = [
    pg.K_PERIOD,    #Forward
    pg.K_e,         #Backward
    pg.K_o,         #Left
    pg.K_u,         #Right
]



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

        return action_to_id[tuple(key_state.values())]

    return controller
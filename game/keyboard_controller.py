import pygame as pg

wasd_control_scheme = [
    pg.K_w,     #Forward
    pg.K_s,     #Backward
    pg.K_a,     #Left
    pg.K_d,     #Right
    pg.K_SPACE, #Shoot
]

uhjk_control_scheme = [
    pg.K_u,     #Forward
    pg.K_j,     #Backward
    pg.K_h,     #Left
    pg.K_k,     #Right
    pg.K_l,     #Shoot
]


def create_keyboard_controller(events_retriever, control_scheme):
    key_state = {k: False for k in control_scheme }

    def controller(simulator, agent_id, time_delta):
        for e in events_retriever():
            if e.type == pg.KEYDOWN:
                if e.key in key_state:
                    key_state[e.key] = True
            elif e.type == pg.KEYUP:
                if e.key in key_state:
                    key_state[e.key] = False

        return tuple(key_state.values())

    return controller
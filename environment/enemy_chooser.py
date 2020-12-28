import random as r

def create_enemy_chooser(enemy_ids):
    #If there are enemies, choose a random one
    if len(enemy_ids):
        enemy_id = r.choice(enemy_ids)
    #Else, set no enemies
    else:
        enemy_id = None


    def chooser(simulator):
        nonlocal enemy_id
        
        #Mark first alive enemy as target, else None
        if enemy_id is None or enemy_id in simulator.dead_agents:
            enemy_id = None
            for id in enemy_ids:
                if id not in simulator.dead_agents:
                    enemy_id = id
                    return simulator.agents[enemy_id]

            return None
        else:
            return simulator.agents[enemy_id]


    return chooser
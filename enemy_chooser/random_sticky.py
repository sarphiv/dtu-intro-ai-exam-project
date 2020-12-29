import random as r

def create_random_sticky(enemy_ids):
    #If there are enemies, choose a random one
    if len(enemy_ids):
        enemy_id = r.choice(enemy_ids)
    #Else, set no enemies
    else:
        enemy_id = None


    def chooser(simulator, agent_id, time_delta):
        nonlocal enemy_id
        
        #Mark first alive enemy as target, else None
        if enemy_id is None or enemy_id in simulator.dead_agents:
            enemy_id = None
            for id in enemy_ids:
                if id not in simulator.dead_agents:
                    enemy_id = id
                    return enemy_id

            return None
        else:
            return enemy_id


    return chooser
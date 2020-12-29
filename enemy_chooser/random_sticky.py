import random as r

from environment.simulator import Simulator


def create_random_sticky(enemy_ids):
    #If there are enemies, choose a random one
    if len(enemy_ids):
        enemy_id = r.choice(enemy_ids)
    #Else, set no enemies
    else:
        enemy_id = None


    def chooser(simulator: Simulator, agent_id, time_delta):
        nonlocal enemy_id
        
        #If no marked alive enemy, mark random alive enemy as target, else None
        if enemy_id is None or enemy_id in simulator.dead_agents:
            #Get all alive enemy IDs
            alive_enemies = { id for id in enemy_ids if id not in simulator.dead_agents }

            #If there are enemies alive, mark a random one
            if len(alive_enemies):
                enemy_id = r.choice(list(alive_enemies))
            #Else, mark no enemy
            else:
                enemy_id = None


        #Return marked enemy, if none, return None
        return enemy_id


    return chooser
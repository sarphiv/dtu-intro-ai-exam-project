def create_first(enemy_ids):
    def chooser(simulator, agent_id, time_delta):

        #Mark first alive enemy as target, else None
        for id in enemy_ids:
            if id not in simulator.dead_agents:
                return id

        return None


    return chooser
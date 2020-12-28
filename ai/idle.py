
def create_idle_controller():
    def controller(simulator, agent_id, time_delta):
        return (
            False, #Forward
            False, #Backward
            False, #Left
            False, #Right
            False  #Shoot
        )

    return controller
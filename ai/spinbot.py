
from environment.simulator import Simulator


def create_spinbot_controller():
    def controller(simulator: Simulator, agent_id, time_delta):
        #Action is to always turn counter-clockwise and shoot
        return (
            False,  #Forward
            False,  #Backward
            True,   #Left
            False,  #Right
            True    #Shoot
        )

    return controller
from shapely.geometry import Polygon

class Simulator(object):
    """
    Runs a simulation of a game
    """

    def __init__(self, agents, controllers, kill_boxes):
        super().__init__()
        
        self.agents = agents
        self.controllers = controllers
        self.kill_boxes = kill_boxes
        self.bullets = []


    def find_losers(self):
        losers = []
        
        for i, agent in enumerate(self.agents):
            agent_rect = Polygon(agent.get_rect())

            for box in self.kill_boxes:
                box_rect = Polygon(box)
                if agent_rect.intersects(box_rect):
                    losers.append(i)
                    break

        return losers


    def update(self, time_delta):
        for controller in self.controllers:
            controller(self, time_delta)
            
        bullets_collided = []
        for i, bullet in enumerate(self.bullets):
            collided = False
            bullet_rect = Polygon(bullet.get_rect())

            for box in self.kill_boxes:
                box_rect = Polygon(box)
                if bullet_rect.intersects(box_rect):
                    collided = True

            for agent in self.agents:
                agent_rect = Polygon(agent.get_rect())
                if agent_rect.intersects(bullet_rect):
                    agent.impact(bullet)
                    collided = True

            if collided:
                bullets_collided.append(i)

        self.bullets = [bullet for i, bullet in enumerate(self.bullets) if i not in bullets_collided]


        for agent in self.agents:
            agent.update(time_delta)

        for bullet in self.bullets:
            bullet.update(time_delta)


        return self.find_losers()

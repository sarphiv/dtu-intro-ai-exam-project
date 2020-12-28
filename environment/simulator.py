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


    def run_controllers(self, time_delta):
        #Save actions
        actions = []
        
        #Iterate through all controllers and save actions
        #NOTE: Not executing actions right away
        # as agents whose controllers are ran later,
        # would get information about earlier agents' future actions.
        for agent_id, controller in enumerate(self.controllers):
            actions.append(controller(self, agent_id, time_delta))

        #Iterate through all actions and execute them
        for agent_id, action in enumerate(actions):
            #Destructure actions
            (forward, backward, left, right, shoot) = action
            
            #Get associated agent
            agent = self.agents[agent_id]


            #Execute actions
            if forward != backward:
                agent.move(forward, time_delta)
                
            if left != right:
                agent.turn(left, time_delta)
                
            if shoot:
                bullet = agent.shoot()
                if bullet is not None:
                    self.bullets.append(bullet)


    def collide_bullets(self):
        #Track bullet indexes that have collided
        bullets_collided = []
        
        #Iterate through all bullets
        for i, bullet in enumerate(self.bullets):
            #Treck if it has collided
            collided = False
            #Get hitbox for bullet
            bullet_rect = Polygon(bullet.get_rect())

            #If bullet has hit a kill box, mark collided
            for box in self.kill_boxes:
                box_rect = Polygon(box)
                if bullet_rect.intersects(box_rect):
                    collided = True

            #If bullet has hit an agent, impact agent and mark collided
            for agent in self.agents:
                agent_rect = Polygon(agent.get_rect())
                if agent_rect.intersects(bullet_rect):
                    agent.impact(bullet)
                    collided = True

            #If bullet has collided, add to list of collided bullets
            if collided:
                bullets_collided.append(i)


        #Remove all collided bullets
        self.bullets = [bullet for i, bullet in enumerate(self.bullets) if i not in bullets_collided]


    def update_agents(self, time_delta):
        for agent in self.agents:
            agent.update(time_delta)

    def update_bullets(self, time_delta):
        for bullet in self.bullets:
            bullet.update(time_delta)


    def find_losers(self):
        #Track losers
        losers = []

        #If an agent is touching a kill box, mark agent ID as losing
        for i, agent in enumerate(self.agents):
            agent_rect = Polygon(agent.get_rect())

            for box in self.kill_boxes:
                box_rect = Polygon(box)
                if agent_rect.intersects(box_rect):
                    losers.append(i)
                    break

        #Return all agents that have lost
        return losers


    def update(self, time_delta):
        #Execute actions
        self.run_controllers(time_delta)
        #Execute bullet colisions
        self.collide_bullets()

        #Update agents and non-collided bullets
        self.update_agents(time_delta)
        self.update_bullets(time_delta)

        #Return losers
        return self.find_losers()

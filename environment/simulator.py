from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

class Simulator(object):
    """
    Runs a simulation of a game
    """

    def __init__(self, agents, controllers, kill_zones):
        super().__init__()
        
        self.agents = agents
        self.controllers = controllers
        self.kill_zones = [ Polygon(box) for box in kill_zones ]
        self.bullets = []
        self.dead_agents = {}


    @property
    def alive_agents(self):
        return {agent_index: agent for agent_index, agent in enumerate(self.agents) if agent_index not in self.dead_agents}
    
    
    def intersects_kill_zone(self, shape: BaseGeometry):
        #Return true if shape intersects a kill zone
        for zone in self.kill_zones:
            if shape.intersects(zone):
                return True

        #Else, return false
        return False


    def run_controllers(self, time_delta):
        #Save actions
        actions = []
        
        #Iterate through all alive agents' controllers and save actions
        #NOTE: Not executing actions right away
        # as agents whose controllers are ran later,
        # would get information about earlier agents' future actions.
        for agent_id, controller in enumerate(self.controllers):
            if agent_id not in self.dead_agents:
                actions.append((agent_id, controller(self, agent_id, time_delta)))

        #Iterate through all alive agents' actions and execute them
        for agent_id, action in actions:
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

            #If bullet has hit a kill zone, mark collided
            collided = self.intersects_kill_zone(bullet_rect)

            #If bullet has hit an agent, impact agent and mark collided
            for agent in self.alive_agents.values():
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
        for agent in self.alive_agents.values():
            agent.update(time_delta)

    def update_bullets(self, time_delta):
        for bullet in self.bullets:
            bullet.update(time_delta)


    def update_agent_kill_zones(self):
        #Track dead agents
        dead_agents = []

        #If an agent is touching a kill zone, mark agent ID as dead
        for i, agent in enumerate(self.agents):
            agent_rect = Polygon(agent.get_rect())

            if self.intersects_kill_zone(agent_rect):
                dead_agents.append(i)
                self.dead_agents[i] = agent

        #Return all agents that have 
        return dead_agents


    def update(self, time_delta):
        #Execute actions
        self.run_controllers(time_delta)
        #Execute bullet colisions
        self.collide_bullets()

        #Update agents and non-collided bullets
        self.update_agents(time_delta)
        self.update_bullets(time_delta)
        
        #Update dead agents
        self.update_agent_kill_zones()

        #Return dead agents
        return [agent_index for agent_index in self.dead_agents.keys()]

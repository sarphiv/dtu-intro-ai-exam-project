class Bullet(object):
    """
    Bullet shot by agents
    """

    def __init__(self, position, velocity, width, damage):
        super().__init__()

        self.position = position
        self.velocity = velocity
        self.width = width
        self.damage = damage

    def get_rect(self):
        """
        Get bounding box for agent
        """
        ((x, y), w) = (self.position, self.width)

        (x1, x2) = (x - w / 2, x + w / 2)
        (y1, y2) = (y - w / 2, y + w / 2)

        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        

    def update(self, time_delta):
        """
        Update position of bullet
        """
        self.position = self.position + self.velocity * time_delta
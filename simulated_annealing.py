import util as ut

class SimulatedAnnealing:

    def __init__(self, name, coordinates, **kwargs):
        self.coordinates = coordinates
        self.name = name
        self.path = []

    def get_node_coordinates(self, node_id):
        return self.coordinates[node_id]

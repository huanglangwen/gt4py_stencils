class Grid:
    def __init__(self, grid_dict):
        self.is_ = grid_dict['is_']
        self.ie = grid_dict['ie']
        self.js = grid_dict['js']
        self.je = grid_dict['je']
        self.nic = grid_dict['nic']
        self.njc = grid_dict['njc']
        self.nid = grid_dict['nid']
        self.njd = grid_dict['njd']
        self.npz = grid_dict['npz']
        self.area_64 = grid_dict['area_64']
        self.area = grid_dict['area']
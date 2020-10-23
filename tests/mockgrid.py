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

    def insert_left_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        var[:i_index, :, :] = edge_data_i
        var[:, :j_index, :] = edge_data_j

    def insert_right_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        var[i_index:, :, :] = edge_data_i
        var[:, j_index:, :] = edge_data_j

    def overwrite_edges(self, var, edgevar, left_i_index, left_j_index):
        self.insert_left_edge(
            var,
            edgevar[:left_i_index, :, :],
            left_i_index,
            edgevar[:, :left_j_index, :],
            left_j_index,
        )
        right_i_index = self.ie + left_i_index
        right_j_index = self.ie + left_j_index
        self.insert_right_edge(
            var,
            edgevar[right_i_index:, :, :],
            right_i_index,
            edgevar[:, right_j_index:, :],
            right_j_index,
        )

    def domain_shape_standard(self):
        return (self.nid, self.njd, self.npz)
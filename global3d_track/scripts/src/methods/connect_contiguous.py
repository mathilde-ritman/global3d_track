'''
Mathilde Ritman, mathilde.ritman@physics.ox.ac.uk 2024

'''

import numpy as np
import cc3d
from scipy import ndimage as ndi
import networkx as nx
from scipy.spatial import cKDTree
import itertools
import scipy.ndimage  as ndi
import xarray as xr



class Connect:

    '''
    This class provides three methods to find and label connected components in a boolean array.

    Example useage:
        from my_library.tobac_tracking.connect_contiguous import Connect
        result = Connect(my_data, method='cc3d').get_components()

    '''

    def __init__(self, boolean_array, method='ndimage'):
        self.boolean_array = boolean_array

        if method == 'cc3d':
            self.method = self.get_components_cc3d
        elif method == 'ndimage':
            self.method = self.get_components_ndimage
        elif method == 'kdtree_nx':
            self.method = self.get_components_kdtree_nx
        else:
            raise ValueError('Method not recognised. Choose from: cc3d, ndimage, kdtree_nx')

    def get_components(self, PBC_flag=None, dims_to_skip=()):
        ''' treat data along dims in dims_to_skip as indpendant, i.e., labels are not shared between these dimensions. '''

        arr = self.boolean_array

        output = np.zeros(arr.shape, dtype=int)
        slices = [slice(None)] * arr.ndim
        ranges = [range(arr.shape[i]) for i in dims_to_skip]
        prev_max = 0
        for idx in itertools.product(*ranges):
            for i, dim in enumerate(dims_to_skip):
                slices[dim] = idx[i]
            sliced_arr = arr[tuple(slices)]
            output[tuple(slices)] = self.method(sliced_arr) + (prev_max * sliced_arr)
            prev_max = np.max(output[tuple(slices)])

        if PBC_flag:
            output = self.periodic_boundary(output, PBC_flag)
        return output

    def periodic_boundary(self, components, PBC_flag="hdim_2", t_dim=0, v_dim=1):

        if components.ndim == 4:
            slices = [slice(None)] * components.ndim
            for t in range(components.shape[t_dim]):
                for v in range(components.shape[v_dim]):
                    slices[t_dim] = t
                    slices[v_dim] = v
                    components[tuple(slices)] = self.apply_periodic_boundary(components[tuple(slices)], PBC_flag)

        elif components.ndim == 3:
            slices = [slice(None)] * components.ndim
            for t in range(components.shape[t_dim]):
                slices[t_dim] = t
                components[tuple(slices)] = self.apply_periodic_boundary(components[tuple(slices)], PBC_flag)

        else:
            components = self.apply_periodic_boundary(components, PBC_flag)

        return components

    def apply_periodic_boundary(self, x, PBC_flag):
        
        dim = int(PBC_flag.split('_')[-1]) - 1
        dims = list(range(x.ndim))
        dims.remove(dim)
        alt_dim = dims[0]

        slices = [slice(None)] * x.ndim
        for i in range(x.shape[alt_dim]):
            slices[alt_dim] = i
            first = slices.copy()
            first[dim] = 0
            last = slices.copy()
            last[dim] = -1
            first, last = tuple(first), tuple(last)

            if x[first] > 0 and x[last] >0:
                x[x == x[last]] = x[first]

        return x

    ## ------ cc3d (connected_components) ------ ##

    def get_components_cc3d(self, boolean_array):
        # not available for dim > 3
        output = cc3d.connected_components(boolean_array)
        return output
    
    ## ------ scipy.ndimage ------ ##

    def get_components_ndimage(self, boolean_array):
        output, n = ndi.label(boolean_array)
        return output

    ## ------ using cKDTree + networkx ------ ##

    def get_components_kdtree_nx(self, boolean_array):
        # from : https://stackoverflow.com/questions/66724201/connected-component-labeling-for-arrays-quasi-images-with-many-dimension
        
        # find neighbours
        coordinates = list(zip(*np.where(boolean_array)))
        tree = cKDTree(coordinates)
        neighbours_by_pixel = tree.query_ball_tree(tree, r=1, p=1) # p=1 -> Manhatten distance; r=1 -> what would be 4-connectivity in 2D

        # create graph and find components
        G = nx.Graph()
        for ii, neighbours in enumerate(neighbours_by_pixel):
            if len(neighbours) > 1:
                G.add_edges_from([(ii, jj) for jj in neighbours[1:]]) # skip first neighbour as that is a self-loop
        components = nx.connected_components(G)

        # create output image
        output = np.zeros_like(boolean_array, dtype=int)
        for ii, component in enumerate(components):
            for idx in component:
                output[coordinates[idx]] = ii+1

        return output
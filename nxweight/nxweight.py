"""
Visualize networks with weighted edges.
"""

__author__ = 'harrigan'

import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt
import networkx as nx


class WeightedGraph:
    def __init__(self, weights, cutoff=1e-10):
        self.weights = weights

        trunc_weights = np.copy(weights)

        # Remove transitions below the cutoff
        if cutoff is not None:
            trunc_weights[trunc_weights < cutoff] = 0.0

        # Ignore self transitions
        diag_inds = np.arange(len(weights))
        trunc_weights[diag_inds, diag_inds] = 0.0

        # Normalize the rest
        trunc_weights = 100 * trunc_weights / np.max(trunc_weights)
        self.trunc_weights = trunc_weights

        # Make a sparse representation
        self.coo_weights = scipy.sparse.coo_matrix(trunc_weights)

        # Make a networkx representation
        self.graph = nx.DiGraph(self.trunc_weights)

        self.position_func = nx.spring_layout


    def plot(self, ax=None, positions=None, scale=1.0, **kwargs):

        # Get axes
        if ax is None:
            ax = plt.gca()

        # Get positions of nodes
        if positions is None:
            layout = self.position_func(self.graph)
            positions = []
            for k in sorted(layout):
                positions += [layout[k]]
            positions = np.asarray(positions)

        # Points
        ax.scatter(positions[:, 0], positions[:, 1], **kwargs)

        # Arrows
        coo = self.coo_weights
        arrow_defaults = dict(arrowstyle='->', connectionstyle='arc3,rad=.2')
        for r, c, dat in zip(coo.row, coo.col, coo.data):
            if r == c:
                # No self transitions pleas
                continue

            # Draw arrows
            x1 = positions[c, 0], positions[c, 1]
            x2 = positions[r, 0], positions[r, 1]
            ax.annotate('', x1, xytext=x2,
                        arrowprops=dict(arrow_defaults, lw=dat / scale))


import numpy as np
import pandas as pd

import isuelogit as isl
from pesuelogit.networks import TransportationNetwork
from mate.models import create_mattwin_model


def toy_network():
    keys = [(0, 1, '0'), (1, 2, '0'), (0, 2, '0')]
    A = isl.factory.NetworkGenerator().generate_adjacency_matrix(links_keys=keys)
    network = TransportationNetwork(A=A, key='toy')
    network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': list(network.links_dict.keys()),
                                                    'alpha': 0.15, 'beta': 4.0,
                                                    'tf': [1.0, 1.0, 2.0], 'k': [100.0, 100.0, 100.0]}))
    Q = np.zeros((3, 3))
    Q[0, 2] = 10
    network.load_OD(Q)
    return network


def build_model(**kwargs):
    model = create_mattwin_model(toy_network(), reference_g=np.array([[10.0, 0.0, 0.0]]), **kwargs)
    model.build()
    return model


def trainable_names(model):
    return [v.name for v in model.trainable_variables]


def test_theta_fixed_by_default():
    model = build_model()
    assert np.allclose(model._theta.numpy(), -1.0)
    assert not model._theta.trainable
    assert 'theta:0' not in trainable_names(model)


def test_theta_trainable_when_requested():
    model = build_model(initial_tt=-2.5, trainable_tt=True)
    assert np.allclose(model._theta.numpy(), -2.5)
    assert model._theta.trainable
    assert 'theta:0' in trainable_names(model)

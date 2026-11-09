import numpy as np
import pytest

from fires.core.genfrb import channel_centres


def test_craft_band_has_336_channels():
    f = channel_centres(751.5, 1087.5, 1.0)
    assert f.size == 336
    assert f[0] == 752.0 and f[-1] == 1087.0
    assert np.allclose(np.diff(f), 1.0)
    assert np.isclose(f.mean(), 919.5)


def test_channels_tile_the_band():
    f = channel_centres(700.0, 800.0, 0.5)
    assert f.size == 200
    assert np.isclose(f[0] - 0.25, 700.0) and np.isclose(f[-1] + 0.25, 800.0)


def test_rejects_partial_channel():
    with pytest.raises(ValueError):
        channel_centres(751.5, 1087.0, 1.0)
    with pytest.raises(ValueError):
        channel_centres(800.0, 700.0, 1.0)

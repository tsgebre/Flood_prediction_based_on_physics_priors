"""Physics-informed flood-prediction package.

Modules
-------
* :mod:`src.dataset` -- data loading, the missingness channel, meteorological
  features, and the degree-day snow model.
* :mod:`src.models`  -- LSTM / GRU with fixed physical constants as buffers.
* :mod:`src.train`   -- training loop and the (precip / snow) physics losses.
* :mod:`src.utils`   -- metrics, reproducible stress testing, and plotting.
"""

from . import dataset, models, train, utils  # noqa: F401

__all__ = ["dataset", "models", "train", "utils"]

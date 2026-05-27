from logging import getLogger

from chitu_diffusion.flexcache.strategy.model import ModelStrategy

logger = getLogger(__name__)


class SeqStrategy(ModelStrategy):
    """V1 sequence-granularity entrypoint.

    The full Wan T2V token-block selective path will grow from this strategy.
    For now it uses the same anchor-cache lifecycle as model granularity and
    keeps a distinct public strategy name so routing and validation are stable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "seq"

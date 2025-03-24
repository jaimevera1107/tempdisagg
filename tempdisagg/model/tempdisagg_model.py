from tempdisagg.model.tempdisagg_core import TempDisaggModelCore
from tempdisagg.model.tempdisagg_ensemble import TempDisaggEnsemble
from tempdisagg.model.tempdisagg_adjuster import TempDisaggAdjuster
from tempdisagg.model.tempdisagg_visualizer import TempDisaggVisualizer
from tempdisagg.model.tempdisagg_summary import TempDisaggReporter


class TempDisaggModel(TempDisaggModelCore,
                      TempDisaggEnsemble,
                      TempDisaggAdjuster,
                      TempDisaggVisualizer,
                      TempDisaggReporter):
    """
    Main orchestration class for temporal disaggregation.

    Combines modular components:
    - TempDisaggModelCore: core logic for initialization and fitting
    - TempDisaggEnsemble: supports ensemble estimation
    - TempDisaggAdjuster: post-estimation adjustment logic
    - TempDisaggVisualizer: plotting functionality
    - TempDisaggReporter: summary, validation, and statistics
    """
    pass

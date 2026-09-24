"""s_res_var_log: inference-time new-s^2 (repaired Section 2.5) diagnostics.

The trained PT-Flow checkpoint is unchanged; this package only re-characterizes
the tilted estimator's variance and sample budget using the chi^2 read directly
from the importance weights, instead of the old log-normal e^{s^2}.
"""

from s_res_var_log.s_res import (  # noqa: F401
    VarianceReport,
    chi2_defensive_upper_bound,
    chi2_from_log_w,
    ess_fraction,
    eps_sweep,
    logw_variance,
    offdiagonal_curvature_term,
    sample_budget,
    variance_report,
)

"""Weight-only warm initialization: source iteration/optimizer/PT state ignored."""
import torch
from utils.ckpt_util import canonical_state_dict, check_model_behavior
from utils.dist_util import unwrap_ddp
from utils.logging import log_for_0


def initialize_ema_only(state, path):
    if int(state.step) != 0 or state.optimizer.state:
        raise ValueError("Weight-only initialization requires step zero and a fresh generator optimizer")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    check_model_behavior(state.model, payload)
    if payload.get("ema_model") is None:
        raise ValueError("init_ema_from requires EMA weights in the source checkpoint")
    weights = canonical_state_dict(payload["ema_model"])
    unwrap_ddp(state.model).load_state_dict(weights, strict=True)
    state.ema_model.load_state_dict(weights, strict=True)
    for ema in state.extra_emas.values():
        ema.load_state_dict(weights, strict=True)
    log_for_0("Initialized EMA weights ONLY from source step %s; NEW training starts at 0; optimizer and PT state fresh", payload.get("step"))
    print(f"WEIGHTS_ONLY_INIT source_step={payload.get('step')} new_step={state.step} optimizer_states={len(state.optimizer.state)}", flush=True)

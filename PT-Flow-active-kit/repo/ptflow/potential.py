"""The PT-Flow data-side potential and its input gradients.

Parameterization
----------------
The paper writes the network output as the log-potential u_theta and argues
(Appendix A) that this is the scale-free choice under eps-annealing.  We output
phi_theta instead, and the reason is load-bearing enough to state here.

    phi_theta has a finite eps -> 0 limit: it converges to the Kantorovich /
    Brenier potential of the unregularized problem, which is fixed by the data.
    u_theta = phi_theta / (2 eps) does not -- it diverges like 1/eps.  Annealing
    eps over two orders of magnitude therefore forces a *u*-parameterized
    network to track a 200x change in its own output scale, while a
    *phi*-parameterized network's target barely moves.

    The magnitudes are not academic.  At latent-ImageNet scale (d = 4096) the
    prox condition 2 eps grad u = x0 - y* with O(1) per-coordinate displacement
    gives |grad u| ~ 1/(2 eps) ~ 500 and u ~ 1e6 at eps = 1e-3.  Every optimizer
    hyperparameter that is not scale-free -- and Adam's epsilon, gradient
    clipping and weight decay are all in that category -- would have to be
    retuned as the anneal proceeds.

This is an argument about the *learning target's scale*, not about float
precision; the precision question is separate and is answered in ptflow/estimator.py
(short version: the binding quantity is phi's offset, which the gauge pin in
ptflow.losses keeps near zero).

output_mode="u" restores the paper-literal parameterization for ablation.

Initialization
--------------
The readout reuses the baseline's FinalLayer, whose output projection is
zero-initialized.  Hence phi_theta == 0 at step 0, so prox_{phi} = identity,
which is exactly the generator's own identity warm-start.  The two networks are
therefore *jointly consistent at initialization*: the proposal is centred on the
true prox of the current potential, the importance weights are uniform, and
ESS/K = 1.  The scheme starts inside the feasible region rather than having to
find its way in.
"""

from __future__ import annotations

from typing import Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn

from models.generator import LightningDiT, TorchLinear
from utils.precision import potential_autocast


class PotentialNet(nn.Module):
    """Scalar potential phi_theta(x, c) on the latent space R^d.

    Args:
        cond_dim: conditioning embedding width.
        num_classes: number of real classes.  One extra embedding index
            (``num_classes``) is allocated as the unconditional token used by
            classifier-free guidance and by conditioning dropout.
        phi_scale: output scale, phi = phi_scale * mean(field).  The natural
            magnitude of phi is O(d) -- it is extensive, a sum of O(1)
            per-coordinate contributions -- so the default d keeps the raw
            network field O(1).
        quad_anchor: tau in the architectural (A2) fallback
            phi(x) = tau/2 |x|^2 + h_theta(x).  Default 0.0 (off).  tau > 0
            breaks the identity warm-start, since prox of tau/2 |x|^2 is
            x/(1+tau), not x.  Turn it on only if the curvature monitor reports
            persistent (A2) violations.
        output_mode: "phi" (default, numerically stable) or "u" (paper-literal).
    """

    def __init__(
        self,
        cond_dim: int,
        num_classes: int = 1000,
        input_size: int = 32,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        n_cls_tokens: int = 0,
        use_bf16: bool = False,
        attn_fp32: bool = True,
        use_remat: bool = False,
        phi_scale: Optional[float] = None,
        quad_anchor: float = 0.0,
        output_mode: str = "phi",
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_classes = int(num_classes)
        self.uncond_index = int(num_classes)  # the null token
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.use_bf16 = bool(use_bf16)
        self.quad_anchor = float(quad_anchor)
        self.output_mode = str(output_mode)
        if self.output_mode not in ("phi", "u"):
            raise ValueError("output_mode must be 'phi' or 'u', got %r" % (output_mode,))

        self.d = self.input_size * self.input_size * self.in_channels
        self.phi_scale = float(phi_scale) if phi_scale is not None else float(self.d)

        self.class_embed = nn.Embedding(self.num_classes + 1, self.cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # LightningDiTBlock discards its cond_dim argument and builds its adaLN
        # modulation as Linear(hidden_size, 6 * hidden_size), so it silently
        # requires cond_dim == hidden_size.  the baseline configs happen to satisfy
        # that (768 == 768); the potential is deliberately narrower than the
        # generator, so it needs an explicit projection instead of inheriting a
        # coincidence.
        self.cond_proj = (
            TorchLinear(self.cond_dim, int(hidden_size), bias=True)
            if self.cond_dim != int(hidden_size)
            else None
        )

        # Scalar-field trunk: out_channels=1 -> [B, H, W, 1], then mean-pooled.
        # FinalLayer's projection is zero-init, so the field -- and hence phi --
        # is identically zero at initialization.
        self.trunk = LightningDiT(
            input_size=self.input_size,
            patch_size=int(patch_size),
            in_channels=self.in_channels,
            hidden_size=int(hidden_size),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            out_channels=1,
            use_qknorm=bool(use_qknorm),
            use_swiglu=bool(use_swiglu),
            use_rope=bool(use_rope),
            use_rmsnorm=bool(use_rmsnorm),
            cond_dim=int(hidden_size),
            n_cls_tokens=int(n_cls_tokens),
            attn_fp32=bool(attn_fp32),
            use_remat=bool(use_remat),
        )

    # -- conditioning -------------------------------------------------------

    def drop_cond(self, c: torch.Tensor, p_uncond: float, generator=None) -> torch.Tensor:
        """Replace labels by the null token with probability ``p_uncond``.

        This is how the conditional and unconditional potentials are trained
        jointly by maximum likelihood against real data -- the only thing
        maximum likelihood is valid for (Appendix F.1).
        """
        if p_uncond <= 0.0:
            return c
        r = torch.rand(c.shape, generator=generator, device=c.device)
        return torch.where(r < float(p_uncond), torch.full_like(c, self.uncond_index), c)

    def null_labels(self, c: torch.Tensor) -> torch.Tensor:
        return torch.full_like(c, self.uncond_index)

    # -- forward ------------------------------------------------------------

    def phi(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Potential in physical units.  x: [B, H, W, C] -> phi: [B] (fp32)."""
        with potential_autocast(x.device, self.use_bf16):
            # Generator samples can be BF16/FP16 while this head has FP32
            # weights. Cast at the boundary without detaching input gradients.
            x = x.to(dtype=self.class_embed.weight.dtype)
            cond = self.class_embed(c)
            if self.cond_proj is not None:
                cond = self.cond_proj(cond)
            field = self.trunk(x, cond, deterministic=True)  # [B, H, W, 1]
        # The readout is forced to fp32 regardless of the trunk's autocast dtype:
        # bf16 carries ~3 decimal digits, and the estimator needs the *spread* of
        # phi across proposal points, which sits many orders below phi itself.
        out = self.phi_scale * field.float().mean(dim=(1, 2, 3))
        if self.quad_anchor != 0.0:
            out = out + 0.5 * self.quad_anchor * x.float().flatten(1).pow(2).sum(dim=1)
        return out

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, eps: Optional[float] = None
    ) -> torch.Tensor:
        """Return phi (output_mode='phi') or u = phi / (2 eps) (output_mode='u')."""
        out = self.phi(x, c)
        if self.output_mode == "u":
            if eps is None:
                raise ValueError("output_mode='u' requires eps")
            return out / (2.0 * float(eps))
        return out


class ScaleNet(nn.Module):
    """The diagonal log-scale head s(x) implementing S in eq. 2.11.

    Why this is a separate module rather than a second head on the generator
    (which is what the paper's "one two-headed network" suggests):

        Checkpoint compatibility is a hard constraint in this port.  A the OT-drift baseline
        state_*.pt at step N must resume under PT-Flow and continue to N+M, so
        the generator's state_dict must stay shape-identical to the release.
        Adding a scale head there would double the final layer's output
        channels and break that permanently.

    Putting it on the theta side is also the more honest placement.  The target
    is S^-1 = I + grad^2 phi(y*), which is a property of the *potential*, not of
    the map -- the paper only attaches it to eta because that made it free.

    It is deliberately small: the target is a smooth, slowly-varying field, and
    it is only ever evaluated at prox points.  ``scale_max`` bounds the output
    through a tanh, because an unbounded proposal width is a direct route to NaN
    in the importance weights.  Zero-init means s == 0 (S = I) at step 0.
    """

    def __init__(
        self,
        cond_dim: int,
        num_classes: int = 1000,
        input_size: int = 32,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        n_cls_tokens: int = 0,
        use_bf16: bool = False,
        attn_fp32: bool = True,
        scale_max: float = 3.0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_classes = int(num_classes)
        self.uncond_index = int(num_classes)
        self.scale_max = float(scale_max)
        self.use_bf16 = bool(use_bf16)

        self.class_embed = nn.Embedding(self.num_classes + 1, self.cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        self.cond_proj = (
            TorchLinear(self.cond_dim, int(hidden_size), bias=True)
            if self.cond_dim != int(hidden_size)
            else None
        )

        self.trunk = LightningDiT(
            input_size=int(input_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            hidden_size=int(hidden_size),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            out_channels=int(in_channels),
            use_qknorm=bool(use_qknorm),
            use_swiglu=bool(use_swiglu),
            use_rope=bool(use_rope),
            use_rmsnorm=bool(use_rmsnorm),
            cond_dim=int(hidden_size),
            n_cls_tokens=int(n_cls_tokens),
            attn_fp32=bool(attn_fp32),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] (a prox point) -> log-scale field [B, H, W, C]."""
        with potential_autocast(x.device, self.use_bf16):
            x = x.to(dtype=self.class_embed.weight.dtype)
            cond = self.class_embed(c)
            if self.cond_proj is not None:
                cond = self.cond_proj(cond)
            raw = self.trunk(x, cond, deterministic=True)
        return self.scale_max * torch.tanh(raw.float() / self.scale_max)


# ---------------------------------------------------------------------------
# Input gradients
# ---------------------------------------------------------------------------

def phi_grad(
    potential: PotentialNet,
    x: torch.Tensor,
    c: torch.Tensor,
    *,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """grad_x phi_theta(x, c).  Returns (grad, phi).

    With create_graph=False (the default) this is a single backward through the
    potential w.r.t. its *input only* -- no double backward, and no gradient
    reaches the caller's parameters.  That is the stable mode: the resulting
    prox target is detached, exactly as the OT-drift baseline detaches its OT ``goal``.

    create_graph=True is the paper-faithful mode: it keeps the graph so that
    d/d_eta of grad_x phi(m_eta) is available, at the cost of a Hessian-vector
    product per step.
    """
    # Differentiate with respect to an FP32 input, rather than rounding the
    # potential gradient back into a low-precision generator-output leaf.
    # In full mode the cast stays connected to the generator for the HVP.
    xin = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    if not (xin.requires_grad and create_graph):
        xin = xin.detach().requires_grad_(True)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    # Fused SDPA kernels do not implement the double backward needed by HVPs.
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH) if create_graph else nullcontext():
        phi = potential.phi(xin, c)
        (g,) = torch.autograd.grad(phi.sum(), xin, create_graph=create_graph)
    return g, (phi if create_graph else phi.detach())


def guided_phi_grad(
    potential: PotentialNet,
    x: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor | float,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Classifier-free-guided potential gradient (eq. 2.20 of the paper).

        grad phi^w = (1 + w) grad phi(x, c) - w grad phi(x, null)

    The conditional and unconditional passes are concatenated into one batch, so
    this costs a single forward/backward through the potential regardless of w.
    """
    # Promote before duplicating the CFG branches so their backward gradients
    # are added in FP32, with a single cast back to the generator's dtype.
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if isinstance(w, (int, float)):
        if float(w) == 0.0:
            g, _ = phi_grad(potential, x, c, create_graph=create_graph)
            return g
        w = torch.full((x.shape[0],), float(w), device=x.device, dtype=torch.float32)

    w = w.to(device=x.device, dtype=torch.float32).reshape(-1)
    if w.numel() == 1:
        w = w.expand(x.shape[0])
    if torch.count_nonzero(w) == 0:
        g, _ = phi_grad(potential, x, c, create_graph=create_graph)
        return g

    x2 = torch.cat([x, x], dim=0)
    c2 = torch.cat([c, potential.null_labels(c)], dim=0)
    g2, _ = phi_grad(potential, x2, c2, create_graph=create_graph)

    b = x.shape[0]
    g_cond, g_uncond = g2[:b], g2[b:]
    wv = w.view(-1, *([1] * (x.ndim - 1)))
    return (1.0 + wv) * g_cond - wv * g_uncond


def prox_residual(
    potential: PotentialNet,
    m: torch.Tensor,
    x0: torch.Tensor,
    c: torch.Tensor,
    w: torch.Tensor | float = 0.0,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """The prox first-order condition residual, in phi-units:

        r = grad phi^w(m) + m - x0

    Written in phi-units (not u-units) so its scale is a transport displacement
    and is therefore invariant under the eps anneal -- Appendix H of the paper.
    """
    g = guided_phi_grad(potential, m, c, w, create_graph=create_graph)
    return g + m - x0

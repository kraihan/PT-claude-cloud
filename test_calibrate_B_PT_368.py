import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

import calibrate_B_PT_368 as cal

ROOT=Path(__file__).resolve().parent
REPO=ROOT/"PT-Flow-active-kit/repo"
sys.path.insert(0,str(REPO))
torch.set_num_threads(1)


def test_scale_fit_recovers_known_quadratic_and_improves_ess():
    from ptflow.estimator import tilted_phi0
    helper=cal.get_helper(ROOT/"audit_B_PT_368.py")
    class Quadratic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.curvature=torch.nn.Parameter(torch.tensor(3.),requires_grad=False)
        def phi(self,x,c):
            return .5*self.curvature*x.flatten(1).square().sum(1)
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.s=torch.nn.Parameter(torch.zeros(1,4,4,4))
        def forward(self,m,c):
            return self.s.expand_as(m)
    torch.manual_seed(9)
    pot,scale=Quadratic(),Scale()
    x=torch.randn(16,4,4,4)
    pairs=dict(x=x,mean=x/4,cond=torch.zeros(16,dtype=torch.long))
    def ess():
        with torch.no_grad():
            est=tilted_phi0(pot,x,pairs["cond"],pairs["mean"],scale(pairs["mean"],pairs["cond"]),
                .1,K=128,alpha_def=0.,generator=torch.Generator().manual_seed(3))
            return float(est.ess.mean())
    old=ess()
    cal.fit_scale(pot,scale,pairs,.1,steps=250,lr=.03,batch=8,K=32)
    assert abs(float(scale.s.mean())+math.log(4))<.05
    assert ess()>.85 and ess()>old+.5
    assert pot.curvature.item()==3. and pot.curvature.grad is None
    pot.curvature.requires_grad_(True)
    with pytest.raises(ValueError,match="frozen"):
        cal.reverse_kl_loss(pot,x,x/4,pairs["cond"],scale.s.expand_as(x),.1)


def test_real_models_scale_only_end_to_end(tmp_path,monkeypatch):
    from models.generator import DitGen
    from ptflow.potential import PotentialNet,ScaleNet
    from ptflow.schedule import build_schedule
    home=tmp_path/"job"
    ckpt=home/"B_PT_Full/checkpoints/state_00000368.pt"
    ckpt.parent.mkdir(parents=True)
    common=dict(cond_dim=8,num_classes=3,input_size=4,in_channels=1,
                hidden_size=8,depth=1,num_heads=2,use_rope=False)
    cfg=dict(model=dict(**common,out_channels=1,patch_size=2,use_bf16=False),
        dataset=dict(num_classes=3),pt=dict(p_uncond=.25,
            model=dict(**common,quad_anchor=1.),scale_model=common,
            schedule=dict(policy="active_recovery_v1",eps_max=.1,eps_min=.05,
                          eps_warmup=20,eps_anneal_steps=20000)))
    (home/"full.yaml").write_text(yaml.safe_dump(cfg))
    gen,pot,scale=DitGen(**cfg["model"]),PotentialNet(**cfg["pt"]["model"]),ScaleNet(**common)
    sched=build_schedule(cfg["pt"]["schedule"])
    for _ in range(368):
        sched.observe(.01)
    torch.save(dict(step=368,model=gen.state_dict(),pt_model=pot.state_dict(),
        pt_scale_model=scale.state_dict(),pt_schedule=sched.state_dict()),ckpt)
    before=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    helper=cal.get_helper(ROOT/"audit_B_PT_368.py")
    monkeypatch.setattr(helper,"validate_source",lambda h:dict(config=home/"full.yaml",repo=REPO,checkpoint=ckpt))
    monkeypatch.setattr(cal,"KS",(16,32))
    monkeypatch.setattr(cal,"MULTIPLIERS",(1.,))
    monkeypatch.setattr(cal,"ALPHAS",(.1,))
    out=tmp_path/"calibration"
    cal.run_calibration(home,out,helper,device_name="cpu",fit_pairs=4,test_pairs=4,
                        fit_steps=2,refine_steps=2,repeats=2)
    report=json.loads((out/"summary.json").read_text())
    assert report["completed"] and report["scale_updated"]
    assert report["generator_unchanged"] and report["potential_unchanged"] and report["schedule_unchanged"]
    assert len(report["primary_before_after"])==4
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==before
    inputs=torch.load(out/"inputs.pt",weights_only=True)
    assert not torch.equal(inputs["fit"]["x"],inputs["test"]["x"])
    candidate=torch.load(out/"scale_candidate.pt",weights_only=False)
    assert "model" not in candidate and "optimizer" not in candidate
    assert any(not torch.equal(v,candidate["proposal_scale_state_dict"][k]) for k,v in scale.state_dict().items())
    assert not (out/"checkpoints").exists()
    with pytest.raises(FileExistsError):
        cal.run_calibration(home,out,helper,device_name="cpu")


def test_decision_does_not_promote_best_test_grid_result():
    rows=[]
    for multiplier in (1.,2.):
        for k in (16,128):
            for cohort in ("conditional","null"):
                rows.append(dict(stage="after",variance_multiplier=multiplier,alpha_def=.1,K=k,
                    cohort=cohort,control_ess_mean=.01 if multiplier==1 else .9,near_one_draw_fraction=0.))
    assert not cal.decision(rows)["candidate_for_extended_pilot"]
    for row in rows:
        row["control_ess_mean"]=.8
    assert cal.decision(rows)["candidate_for_extended_pilot"]
    rows[0]["near_one_draw_fraction"]=.2
    assert not cal.decision(rows)["candidate_for_extended_pilot"]


def test_slurm_worker_bash_syntax(tmp_path):
    path=tmp_path/"worker.sbatch"
    path.write_text(cal.WORKER,newline="\n")
    subprocess.run(["C:/Program Files/Git/bin/bash.exe","-n",str(path)],check=True)
    assert "--gpus=h200:1" in cal.WORKER
    assert "scancel" not in cal.WORKER and "train.py" not in cal.WORKER

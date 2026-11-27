"""B65 fidelity test: imagined (mirror replay) vs real future of the blade tip and of being hit, for a fixed action
(ACT) held for one frame then idle; also the true simulator with hidden hold/identity (_blind) as reference."""
import os,sys,pickle,collections
sys.argv=["x"]; sys.path.insert(0,"/home/user/cog-ai-souls"); os.chdir("/home/user/cog-ai-souls")
exec(open("scripts/mirror_fight.py").read().rsplit("if __name__",1)[0])
from arena3d.mirror import ReplayBoss, anchor, recognise, heading, place
mem,g=pickle.load(open(CK,"rb")); mem.update=False
env=Arena3D(spec(),U,clips(),seed=8,player_strike=pstrike()); env.reset(); mem.reset(env); g.reset(env)
ACT=int(os.environ.get('ACT',0)); rng=np.random.default_rng(0); RE=[];YE=[]; HP=[]; TE=[]; TK=collections.defaultdict(list); K=collections.defaultdict(list); MV=collections.defaultdict(list)
from arena3d.mirror import _yaw_of as _yaw
errs=collections.defaultdict(list); base=collections.defaultdict(list)
for t in range(3000):
    g.observe(env)
    b=env.boss; tau=env.t-mem.onset
    if b.kind=="strike" and mem.label is not None and tau%4==0 and tau<60:
        real=env.clone(1); ims=[]; ok_=mem.label==b.move_id
        rec=recognise(mem,tau,d_now=float(np.linalg.norm(env.p_xy-b.root))) if os.environ.get("RECOG")=="1" else None
        for s in range(3):
            sim=env.clone(2+s)
            if rec:
                i=int(rng.choice(len(rec),p=np.array([r[0] for r in rec]))); _,ep,f_,lab=rec[i]; qn=env.d.qpos[:63]
                o_,fw_=place(ep["q"][f_],qn)
            else:
                ep=mem.sample(mem.label, mem.cur["d0"], rng)
                if ep is None: continue
                f_,o_,fw_=anchor(ep,env.d.qpos[:63].copy(),tau)
            sim.boss=ReplayBoss(b,ep,f_+1,o_,fw_,mem,mem.label,g,rng,env.phase); ims.append(sim)
        e=[];e0=[]; tsims=[]; rh=False; sh=[False]*3; th=[False]*3
        for s2 in range(3):
            ts=env.clone(20+s2); SC._blind(ts, env, np.random.default_rng(s2), 12, True); tsims.append(ts)
        for k in range(15):
            _,_,_,ri=real.step(ACT if k==0 else 0); tr=real.d.site_xpos[real.tip][:2].copy(); rh = rh or ri['hit'] is not None
            for j_,sim in enumerate(ims): _,_,_,si=sim.step(ACT if k==0 else 0); sh[j_] = sh[j_] or si['hit'] is not None; e.append(np.linalg.norm(sim.d.site_xpos[sim.tip][:2]-tr)); K[k].append(e[-1]); MV[spec().program.moves[b.move_id].name].append(e[-1]) if ok_ else None; RE.append(np.linalg.norm(sim.d.qpos[:2]-real.d.qpos[:2])); YE.append(abs((_yaw(sim.d.qpos[3:7])-_yaw(real.d.qpos[3:7])+np.pi)%(2*np.pi)-np.pi))
            e0.append(np.linalg.norm(env.d.site_xpos[env.tip][:2]-tr))
            for j_,ts in enumerate(tsims): _,_,_,ti=ts.step(ACT if k==0 else 0); th[j_] = th[j_] or ti['hit'] is not None; TE.append(np.linalg.norm(ts.d.site_xpos[ts.tip][:2]-tr)); TK[k].append(TE[-1])
        ok = mem.label==b.move_id
        HP.append((rh, np.mean(sh[:len(ims)]) if ims else 0, np.mean(th)))
        errs[ok].append(np.mean(e)); base[ok].append(np.mean(e0))
    dist=np.linalg.norm(b.root-env.p_xy); a=(1 if dist>3 else 5 if dist<2 else 3)*5
    env.step(a); mem.observe(env); env.hp=1e9
for ok in errs: print("label correct" if ok else "label wrong", len(errs[ok]), "replay tip err m", round(np.mean(errs[ok]),2), "| 'tip stays where it is' err", round(np.mean(base[ok]),2))

print('root err',round(np.mean(RE),2),'yaw err deg',round(np.degrees(np.mean(YE)),1), 'median yaw err', round(np.degrees(np.median(YE)),1))
print('by k',[round(np.mean(K[k]),2) for k in range(15)])
print({m:round(np.mean(v),2) for m,v in MV.items()})
print('TRUE-SIM (blind) err',round(np.mean(TE),2),'by k',[round(np.mean(TK[k]),2) for k in range(0,15,2)])

H=np.array(HP,float); y=H[:,0]
from sklearn.metrics import roc_auc_score
print('hit rate',y.mean().round(3),'mirror AUC',round(roc_auc_score(y,H[:,1]),3),'brier',round(((H[:,1]-y)**2).mean(),3),'| true-sim AUC',round(roc_auc_score(y,H[:,2]),3),'brier',round(((H[:,2]-y)**2).mean(),3))

"""Cache top-10 ball candidates (valid/test) of every ball network -> results/exp/cands_<run>.pkl  (GPU, a few minutes)."""
import pickle
import torch
import sn_fb as FB
import sn_lib as S
from sm_data import ROOT

RES = ROOT / "results"
K = 10


def main():
    va, te = S.load("valid"), S.load("test")
    for run in ["ball_r18", "fb_tiny", "fb_r18", "fb_r18_v2"]:
        out = RES / "exp" / f"cands_{run}.pkl"
        if out.exists():
            continue
        if run == "ball_r18":
            m = S.BallNet(pretrained=False).cuda(); m.load_state_dict(torch.load(RES / run / "best.pt")["state"]); m.eval()
            c = {sp: S.ball_candidates(m, d["names"], sp, topk=K) for sp, d in (("valid", va), ("test", te))}
        else:
            ck = torch.load(RES / run / "best.pt", map_location="cuda"); two = ck.get("two_frame", False)
            m = FB.FBNet(ck["backbone"], pretrained=False, in_ch=6 if two else 3).cuda(); m.load_state_dict(ck["state"]); m.eval()
            c = {sp: FB.fb_outputs(m, d["names"], sp, topk=K, prev=FB.prev_names(d) if two else None)[0] for sp, d in (("valid", va), ("test", te))}
        pickle.dump(c, open(out, "wb")); print(run, "done", flush=True)


if __name__ == "__main__":
    main()

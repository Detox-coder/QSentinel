"""
Expanded traffic generators for classifier coverage expansion.
15 new benign classes + 10 new attack classes.
"""
import numpy as np
import pandas as pd
from training.utils import simulate_flow_features

_rng = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════
# BENIGN GENERATORS
# ═══════════════════════════════════════════════════════════════

def gen_discord_text(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        n_msgs = r.integers(5, 40); sizes, iats, dirs = [], [], []
        for m in range(n_msgs):
            sizes.append(r.integers(60, 500)); dirs.append(1)
            if m > 0: iats.append(r.uniform(0.5, 15.0))
            sizes.append(r.integers(40, 200)); dirs.append(-1)
            iats.append(r.uniform(0.02, 0.2))
            if r.random() < 0.4:
                sizes.append(r.integers(80, 600)); dirs.append(-1)
                iats.append(r.uniform(0.3, 8.0))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_discord_text"})
    return pd.DataFrame(rows)

def gen_telegram_text(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        n_msgs = r.integers(3, 30); sizes, iats, dirs = [], [], []
        for m in range(n_msgs):
            sizes.append(r.integers(50, 400)); dirs.append(1)
            if m > 0: iats.append(r.uniform(1.0, 30.0))
            sizes.append(r.integers(40, 250)); dirs.append(-1)
            iats.append(r.uniform(0.05, 0.5))
            if r.random() < 0.3:
                sizes.append(r.integers(60, 350)); dirs.append(-1)
                iats.append(r.uniform(0.5, 12.0))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_telegram_text"})
    return pd.DataFrame(rows)

def gen_github_clone_v2(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(5, 20)):
            sizes.append(r.integers(100, 500)); dirs.append(1 if i%2==0 else -1)
            if i > 0: iats.append(r.uniform(0.01, 0.1))
        for _ in range(r.integers(50, 400)):
            sizes.append(r.integers(800, 1450)); dirs.append(-1)
            iats.append(r.uniform(0.001, 0.01))
            if r.random() < 0.08:
                sizes.append(r.integers(40, 80)); dirs.append(1)
                iats.append(r.uniform(0.005, 0.05))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_github_clone"})
    return pd.DataFrame(rows)

def gen_github_pull_v2(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(3, 10)):
            sizes.append(r.integers(80, 400)); dirs.append(1 if i%2==0 else -1)
            if i > 0: iats.append(r.uniform(0.01, 0.08))
        for _ in range(r.integers(20, 150)):
            sizes.append(r.integers(600, 1400)); dirs.append(-1)
            iats.append(r.uniform(0.001, 0.015))
            if r.random() < 0.1:
                sizes.append(r.integers(40, 80)); dirs.append(1)
                iats.append(r.uniform(0.005, 0.05))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_github_pull"})
    return pd.DataFrame(rows)

def gen_github_push_v2(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(3, 8)):
            sizes.append(r.integers(80, 300)); dirs.append(1 if i%2==0 else -1)
            if i > 0: iats.append(r.uniform(0.01, 0.1))
        for _ in range(r.integers(30, 200)):
            sizes.append(r.integers(800, 1450)); dirs.append(1)
            iats.append(r.uniform(0.001, 0.01))
            if r.random() < 0.08:
                sizes.append(r.integers(40, 80)); dirs.append(-1)
                iats.append(r.uniform(0.005, 0.03))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_github_push"})
    return pd.DataFrame(rows)

def gen_multitab_browse(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for page in range(r.integers(3, 10)):
            for _ in range(r.integers(3, 15)):
                sizes.append(r.integers(100, 400)); dirs.append(1)
                if len(sizes) > 1: iats.append(r.uniform(0.001, 0.05))
                for _ in range(r.integers(1, 5)):
                    sizes.append(r.integers(300, 1450)); dirs.append(-1)
                    iats.append(r.uniform(0.001, 0.015))
            if page < r.integers(2, 9):
                sizes.append(r.integers(20, 60)); dirs.append(1)
                iats.append(r.uniform(1.0, 15.0))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_multitab_browse"})
    return pd.DataFrame(rows)

def gen_mixed_session(n=200, rng=None):
    r = rng or _rng; rows = []
    sub_gens = [gen_multitab_browse, gen_github_pull_v2, gen_discord_text]
    for _ in range(n):
        gen = r.choice(sub_gens)
        df = gen(1, rng=r)
        row = df.iloc[0].to_dict()
        row["label"] = "benign_mixed_session"
        rows.append(row)
    return pd.DataFrame(rows)

def gen_developer_workstation(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for _ in range(r.integers(2, 6)):
            phase = r.choice(["git", "api", "browse"])
            if phase == "git":
                for _ in range(r.integers(10, 50)):
                    sizes.append(r.integers(600, 1400)); dirs.append(-1 if r.random()<0.7 else 1)
                    if len(sizes)>1: iats.append(r.uniform(0.001, 0.02))
            elif phase == "api":
                for _ in range(r.integers(3, 10)):
                    sizes.append(r.integers(100, 600)); dirs.append(1)
                    if len(sizes)>1: iats.append(r.uniform(0.05, 0.5))
                    sizes.append(r.integers(200, 1200)); dirs.append(-1)
                    iats.append(r.uniform(0.01, 0.1))
            else:
                for _ in range(r.integers(5, 15)):
                    sizes.append(r.integers(100, 400)); dirs.append(1)
                    if len(sizes)>1: iats.append(r.uniform(0.01, 0.1))
                    for _ in range(r.integers(1,4)):
                        sizes.append(r.integers(300,1400)); dirs.append(-1)
                        iats.append(r.uniform(0.001,0.02))
            if len(sizes)>1: iats.append(r.uniform(1.0, 10.0))
            sizes.append(r.integers(20,60)); dirs.append(1)
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_developer"})
    return pd.DataFrame(rows)

def gen_package_manager(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(3,8)):
            sizes.append(r.integers(100,400)); dirs.append(1)
            if i>0: iats.append(r.uniform(0.01,0.1))
            sizes.append(r.integers(200,800)); dirs.append(-1)
            iats.append(r.uniform(0.02,0.2))
        for _ in range(r.integers(100,500)):
            sizes.append(r.integers(1200,1450)); dirs.append(-1)
            iats.append(r.uniform(0.0005,0.005))
            if r.random()<0.03:
                sizes.append(r.integers(40,60)); dirs.append(1)
                iats.append(r.uniform(0.01,0.05))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_package_manager"})
    return pd.DataFrame(rows)

def gen_docker_pull(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(3,8)):
            sizes.append(r.integers(100,500)); dirs.append(1)
            if i>0: iats.append(r.uniform(0.01,0.15))
            sizes.append(r.integers(200,1000)); dirs.append(-1)
            iats.append(r.uniform(0.02,0.2))
        n_layers = r.integers(2,6)
        for _ in range(n_layers):
            for _ in range(r.integers(50,200)):
                sizes.append(r.integers(1200,1450)); dirs.append(-1)
                iats.append(r.uniform(0.0005,0.008))
                if r.random()<0.04:
                    sizes.append(r.integers(40,80)); dirs.append(1)
                    iats.append(r.uniform(0.005,0.05))
            iats.append(r.uniform(0.1,1.0))
            sizes.append(r.integers(100,300)); dirs.append(1)
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_docker_pull"})
    return pd.DataFrame(rows)

def gen_docker_push(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(3,8)):
            sizes.append(r.integers(100,500)); dirs.append(1)
            if i>0: iats.append(r.uniform(0.01,0.15))
            sizes.append(r.integers(100,500)); dirs.append(-1)
            iats.append(r.uniform(0.02,0.2))
        n_layers = r.integers(2,5)
        for _ in range(n_layers):
            for _ in range(r.integers(50,200)):
                sizes.append(r.integers(1200,1450)); dirs.append(1)
                iats.append(r.uniform(0.0005,0.008))
                if r.random()<0.05:
                    sizes.append(r.integers(40,80)); dirs.append(-1)
                    iats.append(r.uniform(0.005,0.05))
            iats.append(r.uniform(0.1,0.8))
            sizes.append(r.integers(100,400)); dirs.append(-1)
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_docker_push"})
    return pd.DataFrame(rows)

def gen_cicd_activity(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for _ in range(r.integers(5,20)):
            sizes.append(r.integers(100,600)); dirs.append(1)
            if len(sizes)>1: iats.append(r.uniform(0.05,1.0))
            sizes.append(r.integers(200,1200)); dirs.append(-1)
            iats.append(r.uniform(0.02,0.3))
        if r.random()<0.6:
            for _ in range(r.integers(20,100)):
                sizes.append(r.integers(800,1450)); dirs.append(1 if r.random()<0.5 else -1)
                iats.append(r.uniform(0.001,0.02))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_cicd"})
    return pd.DataFrame(rows)

def gen_cloud_shell(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        n_cmds = r.integers(10,60); sizes, iats, dirs = [], [], []
        for c in range(n_cmds):
            sizes.append(r.integers(40,200)); dirs.append(1)
            if c>0: iats.append(r.uniform(0.3,8.0))
            for _ in range(r.integers(1,5)):
                sizes.append(r.integers(60,800)); dirs.append(-1)
                iats.append(r.uniform(0.01,0.2))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_cloud_shell"})
    return pd.DataFrame(rows)

def gen_vscode_extensions(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        n_exts = r.integers(2,8)
        for _ in range(n_exts):
            sizes.append(r.integers(100,400)); dirs.append(1)
            if len(sizes)>1: iats.append(r.uniform(0.05,0.5))
            for _ in range(r.integers(10,80)):
                sizes.append(r.integers(800,1450)); dirs.append(-1)
                iats.append(r.uniform(0.001,0.01))
                if r.random()<0.05:
                    sizes.append(r.integers(40,80)); dirs.append(1)
                    iats.append(r.uniform(0.005,0.05))
            iats.append(r.uniform(0.2,2.0))
            sizes.append(r.integers(40,100)); dirs.append(1)
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_vscode_ext"})
    return pd.DataFrame(rows)

def gen_background_sync(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        n_syncs = r.integers(5,25); sizes, iats, dirs = [], [], []
        for s in range(n_syncs):
            sizes.append(r.integers(40,200)); dirs.append(1)
            if s>0: iats.append(r.uniform(2.0,30.0))
            sizes.append(r.integers(40,300)); dirs.append(-1)
            iats.append(r.uniform(0.05,0.5))
            if r.random()<0.2:
                for _ in range(r.integers(2,8)):
                    sizes.append(r.integers(200,1000)); dirs.append(-1 if r.random()<0.6 else 1)
                    iats.append(r.uniform(0.01,0.1))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "benign_background_sync"})
    return pd.DataFrame(rows)

EXPANDED_BENIGN_GENERATORS = [
    gen_discord_text, gen_telegram_text, gen_github_clone_v2, gen_github_pull_v2,
    gen_github_push_v2, gen_multitab_browse, gen_mixed_session, gen_developer_workstation,
    gen_package_manager, gen_docker_pull, gen_docker_push, gen_cicd_activity,
    gen_cloud_shell, gen_vscode_extensions, gen_background_sync,
]


# ═══════════════════════════════════════════════════════════════
# ATTACK GENERATORS
# ═══════════════════════════════════════════════════════════════

def gen_hybrid_c2_exfil(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        # C2 beacon phase
        for i in range(r.integers(8,25)):
            sizes.append(r.integers(30,180)); dirs.append(1)
            if i>0: iats.append(r.uniform(0.5,4.0)*(1+r.uniform(-0.2,0.2)))
            if r.random()<0.4:
                sizes.append(r.integers(40,400)); dirs.append(-1)
                iats.append(r.uniform(0.01,0.05))
        # Transition pause
        iats.append(r.uniform(1.0,5.0)); sizes.append(r.integers(30,80)); dirs.append(1)
        # Exfil phase
        total = r.integers(50000,300000); sent=0
        while sent<total:
            chunk=min(r.integers(400,1400),total-sent)
            sizes.append(chunk); dirs.append(1); iats.append(r.uniform(0.001,0.015)); sent+=chunk
            if r.random()<0.12:
                sizes.append(r.integers(40,80)); dirs.append(-1); iats.append(r.uniform(0.002,0.02))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_hybrid_c2_exfil"})
    return pd.DataFrame(rows)

def gen_adaptive_exfil(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        rate = r.uniform(0.005,0.02); total=r.integers(30000,200000); sent=0
        while sent<total:
            chunk = r.integers(200,1200)
            sizes.append(min(chunk,total-sent)); dirs.append(1); sent+=chunk
            cur_rate = rate * r.uniform(0.5,2.0)
            if len(sizes)>1: iats.append(cur_rate)
            if r.random()<0.2:
                sizes.append(r.integers(200,1000)); dirs.append(-1); iats.append(r.uniform(0.01,0.05))
            if r.random()<0.05:
                iats.append(r.uniform(0.5,3.0)); sizes.append(r.integers(40,100)); dirs.append(1)
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_adaptive_exfil"})
    return pd.DataFrame(rows)

def gen_low_rate_exfil(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        n_chunks=r.integers(15,60); sizes, iats, dirs = [], [], []
        for i in range(n_chunks):
            sizes.append(r.integers(40,300)); dirs.append(1)
            if i>0: iats.append(r.uniform(2.0,30.0))
            sizes.append(r.integers(30,80)); dirs.append(-1); iats.append(r.uniform(0.05,0.3))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_low_rate_exfil"})
    return pd.DataFrame(rows)

def gen_dev_mimic_exfil(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for i in range(r.integers(3,8)):
            sizes.append(r.integers(80,300)); dirs.append(1 if i%2==0 else -1)
            if i>0: iats.append(r.uniform(0.01,0.1))
        total=r.integers(100000,500000); sent=0
        while sent<total:
            chunk=min(r.integers(800,1450),total-sent)
            sizes.append(chunk); dirs.append(1); iats.append(r.uniform(0.001,0.01)); sent+=chunk
            if r.random()<0.06:
                sizes.append(r.integers(40,80)); dirs.append(-1); iats.append(r.uniform(0.005,0.03))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_dev_mimic_exfil"})
    return pd.DataFrame(rows)

def gen_browser_mimic_exfil(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for _ in range(r.integers(5,20)):
            sizes.append(r.integers(200,800)); dirs.append(1)
            if len(sizes)>1: iats.append(r.uniform(0.05,0.5))
            sizes.append(r.integers(100,400)); dirs.append(-1); iats.append(r.uniform(0.02,0.1))
            if r.random()<0.4:
                for _ in range(r.integers(2,8)):
                    sizes.append(r.integers(600,1400)); dirs.append(1); iats.append(r.uniform(0.002,0.02))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_browser_mimic_exfil"})
    return pd.DataFrame(rows)

def gen_interactive_tunnel(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        n_cmds=r.integers(15,80); sizes, iats, dirs = [], [], []
        for c in range(n_cmds):
            sizes.append(r.integers(30,150)); dirs.append(1)
            if c>0: iats.append(r.uniform(0.1,5.0))
            for _ in range(r.integers(1,8)):
                sizes.append(r.integers(40,1200)); dirs.append(-1); iats.append(r.uniform(0.005,0.1))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_interactive_tunnel"})
    return pd.DataFrame(rows)

def gen_cloud_storage_abuse(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        n_files = r.integers(3,10)
        for _ in range(n_files):
            for i in range(r.integers(3,6)):
                sizes.append(r.integers(100,400)); dirs.append(1)
                if len(sizes)>1: iats.append(r.uniform(0.01,0.1))
                sizes.append(r.integers(100,400)); dirs.append(-1); iats.append(r.uniform(0.02,0.2))
            total=r.integers(20000,200000); sent=0
            while sent<total:
                chunk=min(r.integers(1000,1450),total-sent)
                sizes.append(chunk); dirs.append(1); iats.append(r.uniform(0.0005,0.008)); sent+=chunk
                if r.random()<0.04:
                    sizes.append(r.integers(40,80)); dirs.append(-1); iats.append(r.uniform(0.01,0.05))
            iats.append(r.uniform(0.5,3.0)); sizes.append(r.integers(40,100)); dirs.append(1)
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_cloud_abuse"})
    return pd.DataFrame(rows)

def gen_staged_exfil(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        n_stages = r.integers(3,8)
        for _ in range(n_stages):
            for _ in range(r.integers(5,20)):
                sizes.append(r.integers(200,1200)); dirs.append(1); 
                if len(sizes)>1: iats.append(r.uniform(0.005,0.05))
                if r.random()<0.3:
                    sizes.append(r.integers(100,600)); dirs.append(-1); iats.append(r.uniform(0.01,0.05))
            for _ in range(r.integers(3,10)):
                sizes.append(r.integers(100,500)); dirs.append(1 if r.random()<0.5 else -1)
                iats.append(r.uniform(0.5,5.0))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_staged_exfil"})
    return pd.DataFrame(rows)

def gen_beacon_tunnel(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        interval = r.uniform(0.3,3.0)
        for i in range(r.integers(5,15)):
            sizes.append(r.integers(30,150)); dirs.append(1)
            if i>0: iats.append(interval*(1+r.uniform(-0.15,0.15)))
            if r.random()<0.3:
                sizes.append(r.integers(40,200)); dirs.append(-1); iats.append(r.uniform(0.01,0.05))
        iats.append(r.uniform(0.5,2.0)); sizes.append(r.integers(40,100)); dirs.append(1)
        for _ in range(r.integers(20,80)):
            sizes.append(r.integers(100,1400)); dirs.append(1)
            iats.append(r.uniform(0.005,0.05))
            sizes.append(r.integers(100,1400)); dirs.append(-1)
            iats.append(r.uniform(0.005,0.08))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_beacon_tunnel"})
    return pd.DataFrame(rows)

def gen_multi_phase_attack(n=200, rng=None):
    r = rng or _rng; rows = []
    for _ in range(n):
        sizes, iats, dirs = [], [], []
        for _ in range(r.integers(3,8)):
            sizes.append(r.integers(100,500)); dirs.append(1)
            if len(sizes)>1: iats.append(r.uniform(0.1,1.0))
            sizes.append(r.integers(200,1000)); dirs.append(-1); iats.append(r.uniform(0.02,0.2))
        for i in range(r.integers(5,15)):
            sizes.append(r.integers(30,120)); dirs.append(1)
            iats.append(r.uniform(1.0,5.0))
            if r.random()<0.3:
                sizes.append(r.integers(40,300)); dirs.append(-1); iats.append(r.uniform(0.01,0.05))
        total=r.integers(30000,200000); sent=0
        while sent<total:
            chunk=min(r.integers(300,1400),total-sent)
            sizes.append(chunk); dirs.append(1); iats.append(r.uniform(0.002,0.02)); sent+=chunk
            if r.random()<0.1:
                sizes.append(r.integers(40,100)); dirs.append(-1); iats.append(r.uniform(0.005,0.03))
        rows.append({**simulate_flow_features(sizes, iats[:len(sizes)-1], dirs), "label": "malicious_multi_phase"})
    return pd.DataFrame(rows)

EXPANDED_ATTACK_GENERATORS = [
    gen_hybrid_c2_exfil, gen_adaptive_exfil, gen_low_rate_exfil,
    gen_dev_mimic_exfil, gen_browser_mimic_exfil, gen_interactive_tunnel,
    gen_cloud_storage_abuse, gen_staged_exfil, gen_beacon_tunnel, gen_multi_phase_attack,
]

ALL_EXPANDED_GENERATORS = EXPANDED_BENIGN_GENERATORS + EXPANDED_ATTACK_GENERATORS

def generate_expanded(n_per_class=200, seed=42):
    rng = np.random.default_rng(seed)
    frames = [gen(n_per_class, rng=rng) for gen in ALL_EXPANDED_GENERATORS]
    result = pd.concat(frames, ignore_index=True)
    print(f"  Generated {len(result)} expanded flows:")
    for l, c in result["label"].value_counts().items():
        print(f"    {l}: {c}")
    return result

if __name__ == "__main__":
    df = generate_expanded(50)
    print(f"\nTotal: {len(df)} flows, {df['label'].nunique()} classes")

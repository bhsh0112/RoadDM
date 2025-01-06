python3 step0_transfer.py --mode traj
python3 step0_transfer.py --mode eta_task
python3 step1_mapMatching.py --mode traj
python3 step1_mapMatching.py --mode eta_task
python3 step1_mapMatching.py --mode traj --denoise
python3 step2_roadClassify.py
python3 step3_etaEst.py
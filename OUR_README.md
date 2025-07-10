# Autonomous Drone Racing Project Course
## mpcc trajectory translation
### Level2 (stable): success rate 91% average 6.05s
    git checkout main
    python ./scripts/sim.py --config level2.toml --controller mpcc_traj_translation_simplified.py --n_runs 10 -g
### Level2 (leader board): success rate 48% average 4.70s
    git checkout HEAD~1
    python ./scripts/sim.py --config level2.toml --controller mpcc_traj_translation_simplified.py --n_runs 10 -g
## vanilla mpcc
### Level0: average 3.46s
    python ./scripts/sim.py --config level0.toml --controller mpcc.py --n_runs 10 -g
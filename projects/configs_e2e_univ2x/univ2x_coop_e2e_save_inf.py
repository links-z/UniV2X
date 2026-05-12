_base_ = ['./univ2x_coop_e2e.py']

# Step 1: run cooperative inference and save infrastructure track_instances to disk
model_other_agent_inf = dict(
    save_track_query=True,
    save_track_query_file_root='./inf_queries',
)

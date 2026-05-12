_base_ = ['./univ2x_coop_e2e.py']

# Step 2: run cooperative inference loading infrastructure track_instances from disk
model_ego_agent = dict(
    read_track_query_file_root='./inf_queries',
)

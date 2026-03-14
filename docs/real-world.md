# Real-world Franka

## preprocessing
- Here we only provide an example
```bash
# preprocess the real aloha dataset

# pickle generation
python tools/real_world/teledata_process.py
```

## Training
```bash
#planner training 
bash scripts/planner/train_video_1node_robot.sh

# action policy
bash scripts/real_world/train_real_world_robot.sh

# inference
python tools/real_world/real_experiments_server.py
```

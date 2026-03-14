import numpy as np
from transformers import AutoProcessor
import pickle
from tqdm import tqdm

# def split_action_into_subsegments(action, T):
#     N = len(action)
#     num_segments = N // T  # Ignore the trailing segment shorter than T
#     trimmed = action[:num_segments * T]  # Trim extra frames
#     subsegments = trimmed.reshape(num_segments, T, -1)
#     return subsegments

def split_action_into_subsegments(action, T):
    """
    Split the action matrix into contiguous segments of length T, where each
    segment has shape (T, 7). The sliding window stride is 1.
    """
    N = len(action)
    subsegments = []
    
    # Sliding window over action segments
    for start in range(N - T + 1):
        subsegment = action[start:start+T]  # Take T consecutive rows
        subsegments.append(subsegment)
    
    return np.array(subsegments)


# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("/remote-home/jinminghao/structvla/pretrain/fast", trust_remote_code=True)

bridge_pickle = '/remote-home/jinminghao/structvla/datasets/processed_data/meta/simplerenv_bridge_trainval.pkl'
with open(bridge_pickle, 'rb') as f:
    data = pickle.load(f)
##### configure the parameters
T = 10
scale = 50
save_path = '/remote-home/jinminghao/structvla/pretrain/fast_bridge_t5_s50_mytest0908'
#####
all_subsegments = []
for value in tqdm(data):
    action = value["action"]
    subsegments = split_action_into_subsegments(action, T)
    all_subsegments.append(subsegments)

all_subsegments = np.concatenate(all_subsegments, axis=0)

print(all_subsegments.shape)  # Output shape

# test original the tokenizer
tokens = tokenizer(all_subsegments)
decoded_actions = tokenizer.decode(tokens)

# compute the difference between the original and the new decoded actions
diff = np.abs(all_subsegments - decoded_actions)
# mean difference
mean_diff = np.mean(diff)
print("mean diff:", mean_diff)

# train the tokenizer
tokenizer = tokenizer.fit(all_subsegments, scale=scale)
# save the tokenizer
tokenizer.save_pretrained(save_path)

# compute the difference between the original and the new decoded actions
tokens = tokenizer(all_subsegments)
# print average length of the tokens
print(np.mean([len(token) for token in tokens]))
print(np.max([len(token) for token in tokens]))
print(np.min([len(token) for token in tokens]))
decoded_actions = tokenizer.decode(tokens)
mean_diff = np.mean(np.abs(all_subsegments - decoded_actions))
print(mean_diff)




import struct

def read_gamestate(filename):
    with open(filename, "rb") as f:
        states_size = struct.unpack("Q", f.read(8))[0]
        policies_size = struct.unpack("Q", f.read(8))[0]
        values_size = struct.unpack("Q", f.read(8))[0]
        
        states = []
        for _ in range(states_size):
            state_size = struct.unpack("Q", f.read(8))[0]
            state = struct.unpack(f"{state_size}f", f.read(state_size * 4))
            states.append(state)
        
        policies = []
        for _ in range(policies_size):
            policy_size = struct.unpack("Q", f.read(8))[0]
            policy = struct.unpack(f"{policy_size}f", f.read(policy_size * 4))
            policies.append(policy)
        
        values = struct.unpack(f"{values_size}f", f.read(values_size * 4))
    
    return states, policies, values

if __name__ == "__main__":
    gamestate = read_gamestate("dataset_cpu0_0_2025-03-29.pkl")
    print(gamestate)
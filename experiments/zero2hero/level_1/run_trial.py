def run_trial(policy, env, memory):
    initial_state = env.reset()
    state = initial_state
    done = False
    while not done:
        action = policy.compute_action(state, env)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1.0
        memory.record(action=action, reward=reward, state=state)
        state = next_state

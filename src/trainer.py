from purejaxrl.ppo_rnn import make_train
from models.arelit import BatchedAReLiT
from models.gru import BatchedGRU
from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper

import jax
import gymnax
import sys

sys.path.append(".")


if __name__ == "__main__":
    # RNN config for GRU
    rnn_config = {
        "RNN_TYPE": 'gru',
        "HIDDEN_SIZE": 128,
    }

    # RNN config for AReLiT
    # rnn_config = {
    #     "RNN_TYPE": 'arelit',
    #     "N_LAYERS": 2,
    #     "D_MODEL": 128,
    #     "D_HEAD": 64,
    #     "D_FFC": 64,
    #     "N_HEADS": 4,
    #     "ETA": 4,
    #     "R": 2,
    # }

    train_config = {
        "RNN": rnn_config,
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "MemoryChain-bsuite",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "OPTIMISTIC_RESETS": False,
        "HIDDEN": 128,
    }

    if train_config['RNN']['RNN_TYPE'] == "gru":
        def rnn_module():
            return BatchedGRU()

        def rnn_carry_init(batch_size):
            return BatchedGRU.initialize_carry(batch_size, train_config['RNN']['HIDDEN_SIZE'])
    elif train_config['RNN']['RNN_TYPE'] == "arelit":
        def rnn_module():
            return BatchedAReLiT(n_layers=train_config['RNN']['N_LAYERS'],
                                 d_model=train_config['RNN']['D_MODEL'],
                                 d_head=train_config['RNN']['D_HEAD'],
                                 d_ffc=train_config['RNN']['D_FFC'],
                                 n_heads=train_config['RNN']['N_HEADS'],
                                 eta=train_config['RNN']['ETA'],
                                 r=train_config['RNN']['R'])

        def rnn_carry_init(batch_size):
            return BatchedAReLiT.initialize_carry(batch_size, n_layers=2, n_heads=4, d_head=64,
                                                  eta=4, r=2)
    # Initialize the env
    if train_config["ENV_NAME"] == "craftax":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        env = CraftaxSymbolicEnv()
        env_params = env.default_params
    else:
        env, env_params = gymnax.make(train_config["ENV_NAME"])
        env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(train_config, env,
                                   env_params, rnn_module, rnn_carry_init))
    out = train_jit(rng)

import random
import torch
import numpy as np

class NstepReplaySubMemCell(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size

        self.actions = [None] * self.memory_size
        self.rewards = [None] * self.memory_size
        self.states = [None] * self.memory_size
        self.s_primes = [None] * self.memory_size
        self.terminals = [None] * self.memory_size

        self.count = 0
        self.current = 0

    def add(self, s_t, a_t, r_t, s_prime, terminal):
        
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t
        self.s_primes[self.current] = s_prime
        self.terminals[self.current] = terminal

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term):
        for i in range(len(list_st)):
            if list_sp is None:
                sp = (None, None, None)
            else:
                sp = list_sp[i]
            self.add(list_st[i], list_at[i], list_rt[i], sp, list_term[i])

    def sample(self, batch_size):
        assert self.count >= batch_size

        list_st = []
        list_at = []
        list_rt = []
        list_s_primes = []
        list_term = []
        
        for i in range(batch_size):
            idx = random.randint(0, self.count - 1)
            list_st.append(self.states[idx])
            list_at.append(self.actions[idx])
            list_rt.append(self.rewards[idx])
            list_s_primes.append(self.s_primes[idx])
            list_term.append(self.terminals[idx])

        return list_st, list_at, list_rt, list_s_primes, list_term

    def sample_all(self):
        return self.actions[:self.count], self.rewards[:self.count], self.states[:self.count], self.s_primes[:self.count], self.terminals[:self.count]

def hash_state_action(s_t, a_t):
    key = s_t[0]
    base = 179424673
    for e in s_t[1].directed_edges:
        key = (key * base + e[0]) % base
        key = (key * base + e[1]) % base
    if s_t[2] is not None:
        key = (key * base + s_t[2]) % base
    else:
        key = (key * base) % base
    
    key = (key * base + a_t) % base
    return key

class NstepReplayMemCell(object):
    def __init__(self, memory_size, balance_sample=False):
        self.sub_list = []
        self.balance_sample = balance_sample
        self.sub_list.append(NstepReplaySubMemCell(memory_size))
        if balance_sample:
            self.sub_list.append(NstepReplaySubMemCell(memory_size))
            self.state_set = set()

    def add_list(self, s_t, list_a_t, list_r_t, s_prime, terminal):
        self.sub_list[0].add(s_t, list_a_t, list_r_t, s_prime, terminal)
    
    def sample(self, batch_size):
        return self.sub_list[0].sample(batch_size)
        
    def sample_all(self):
        return self.sub_list[0].sample_all()

    def size(self):
        return self.sub_list[0].count

class MQLMem(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size

        self.actions = [None] * self.memory_size
        self.rewards = [None] * self.memory_size
        self.states = [None] * self.memory_size
        self.s_primes = [None] * self.memory_size
        self.terminals = [None] * self.memory_size
        self.scores = [None] * self.memory_size

        self.count = 0
        self.current = 0

    def add(self, s_t, a_t, r_t, s_prime, terminal, score):
        
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t
        self.s_primes[self.current] = s_prime
        self.terminals[self.current] = terminal
        self.scores[self.current] = score

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_list(self, list_st, list_at, list_rt, list_sp, list_term, list_score):
        n = len(list_st)
        for i in range(n):
            self.add(list_st[i], list_at[i], list_rt[i], list_sp[i], list_term[i], list_score[i])

    def prior_sample(self, batch_size):
        assert self.count >= batch_size
        sample_prob = torch.FloatTensor(self.scores[:self.count])
        idxs = torch.multinomial(sample_prob, num_samples=batch_size)

        list_st = []
        list_at = []
        list_rt = []
        list_s_primes = []
        list_term = []
        list_score = []
        for i in range(batch_size):
            idx = idxs[i]

            list_st.append(self.states[idx])
            list_at.append(self.actions[idx])
            list_rt.append(self.rewards[idx])
            list_s_primes.append(self.s_primes[idx])
            list_term.append(self.terminals[idx])
            list_score.append(self.scores[idx])

        return list_st, list_at, list_rt, list_s_primes, list_term, list_score

    def sample(self, batch_size):
        assert self.count >= batch_size

        list_st = []
        list_at = []
        list_rt = []
        list_s_primes = []
        list_term = []
        list_score = []
        
        for i in range(batch_size):
            idx = random.randint(0, self.count - 1)
            list_st.append(self.states[idx])
            list_at.append(self.actions[idx])
            list_rt.append(self.rewards[idx])
            list_s_primes.append(self.s_primes[idx])
            list_term.append(self.terminals[idx])
            list_score.append(self.scores[idx])

        return list_st, list_at, list_rt, list_s_primes, list_term, list_score

    def sample_all(self):
        return self.actions[:self.count], self.rewards[:self.count], self.states[:self.count], self.s_primes[:self.count], self.terminals[:self.count], self.scores[:self.count]


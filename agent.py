import os
import sys
import time
import numpy as np
import torch
import networkx as nx
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import Env
from qnet import QNet
from player import Player
from dataset import GraphLoader
from mem import NstepReplayMemCell
from utils.config import parse_args
from utils.utils import Tee, greedy_actions, uniform_actions, print_res, print_rl_params

class Agent(object):
    def __init__(self, args, env, G, tgt_G, save_path):
        self.args = args
        self.G = G
        self.tgt_G = tgt_G
        
        self.mem_pool = NstepReplayMemCell(memory_size=500000, balance_sample=False)
        self.env = env 
        self.save_path = save_path
        self.model_name = save_path.split('/')[-1]
        print(self.model_name)
               
        self.net = QNet(args, env.statedim)
        self.old_net = QNet(args, env.statedim)
        
        self.net = self.net.cuda()
        self.old_net = self.old_net.cuda()

        self.old_net.eval()

        self.net.reset_parameters()
        self.old_net.reset_parameters()

        self.budget = self.args.query_init + self.args.query_cs * self.args.query_bs
        
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 10000
        self.burn_in = 10  
        self.step = 0        
        self.pos = 0
        self.best_eval = 0.0
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, greedy=False, dataset='source'):
        if dataset == 'source':
            pool = self.env.player.get_pool()
            adj_list = self.G.adj_list
        else:
            pool = self.env.tgt_player.get_pool()
            adj_list = self.tgt_G.adj_list

        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)
        
        cur_state = self.env.get_state(dataset).detach()
        values = self.net(cur_state, adj_list).detach()

        if random.random() < self.eps and not greedy:
            actions = uniform_actions(pool, self.args.query_bs)
        else:
            actions = greedy_actions(pool, values, self.args.query_bs)
        actions = torch.LongTensor(actions).view(-1)
        q_values = values[actions]
        
        return cur_state, actions, q_values

    def run_simulation(self):
        self.env.reset()
        now_acc, now_macro, now_micro = self.env.init_train()

        for i in range(self.args.query_cs):
            state_t, action_t, value_t = self.make_actions(greedy=False, dataset='source')
            
            self.env.player.model_reset()
            acc, macro_f1, micro_f1 = self.env.step(action_t)
            
            reward_t = torch.Tensor([(acc - now_acc)])
            now_acc = acc
            
            if i == self.args.query_cs-1:
                is_terminal = True
                s_prime = None
            else:
                is_terminal = False
                s_prime = (self.env.get_state().clone(), self.env.player.trainmask+self.env.player.testmask+self.env.player.valmask)

            self.mem_pool.add_list(state_t, action_t, reward_t, s_prime, is_terminal)
            
    def eval(self, times=30):
        avg_acc, avg_macro, avg_micro = np.zeros(self.args.query_cs), np.zeros(self.args.query_cs), np.zeros(self.args.query_cs)
        for cs in range(times):
            self.env.reset(dataset='target')
            self.env.init_train(dataset='target')

            accs, macro_f1s, micro_f1s = [], [], []
            for i in range(self.args.query_cs):
                if torch.sum(self.env.tgt_player.trainmask).item() != i * self.args.query_bs + self.args.query_init:
                    print("error!")
                    break
                state_t, action_t, value_t = self.make_actions(greedy=True, dataset='target')
                self.env.tgt_player.model_reset()
                acc, macro_f1, micro_f1 = self.env.step(action_t, dataset='target')
                
                accs.append(acc)
                macro_f1s.append(macro_f1)
                micro_f1s.append(micro_f1)
            
            accs, macro_f1s, micro_f1s = np.array(accs), np.array(macro_f1s), np.array(micro_f1s)
            avg_acc += accs
            avg_macro += macro_f1s
            avg_micro += micro_f1s

        avg_acc /= times
        avg_macro /= times
        avg_micro /= times
        print_res(avg_acc, avg_macro, avg_micro)
        
        eval_acc = np.sum(avg_acc)
        if eval_acc > self.best_eval:
            self.best_eval = eval_acc
            print("Best Result!!")
            # torch.save(self.net.state_dict(), self.model_save_path)

    def save_transitions(self, step=0):
        a, r, s, sprime, terminal = self.mem_pool.sample_all()
        total = len(a)
        actions, rewards, states, sprimes, masks = [], [], [], [], []
        for i in range(total):
            actions.append(a[i].cpu().unsqueeze(0))
            rewards.append(r[i].cpu().unsqueeze(0))
            states.append(s[i].cpu().unsqueeze(0))
            if terminal[i] is True:
                sprimes.append(torch.zeros(s[i].cpu().shape).unsqueeze(0))
                masks.append(torch.zeros(s[i].cpu().shape[0]).unsqueeze(0))
            else:
                sprimes.append(sprime[i][0].cpu().unsqueeze(0))
                masks.append(sprime[i][1].cpu().unsqueeze(0))

        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        states = torch.cat(states, dim=0)
        sprimes = torch.cat(sprimes, dim=0)
        masks = torch.cat(masks, dim=0)
        # print(actions.shape, rewards.shape, states.shape, sprimes.shape, masks.shape)
        save_info = {}
        save_info['action'] = actions
        save_info['reward'] = rewards
        save_info['state'] = states
        save_info['sprime'] = sprimes
        save_info['masks'] = masks
        save_transition_path = os.path.join(self.save_path, '{}_tran.pt'.format(self.model_name))
        torch.save(save_info, save_transition_path)
    
    def train(self):
        for p in range(self.burn_in):
            self.run_simulation()
            
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.rllr)
        tf = time.time()

        for self.step in range(self.args.rl_epoch):
            self.run_simulation()
            
            if (self.step+1) % self.args.rl_update == 0:
                self.take_snapshot()
            if (self.step+1) % 50 == 0:
                save_model_path = os.path.join(self.save_path, '{}.model'.format(self.model_name))
                torch.save(self.net.state_dict(), save_model_path)
                self.save_transitions(self.step)
                # self.eval()
                
            for t in range(self.args.rl_train_iters):
                list_states, list_actions, list_rewards, list_s_primes, list_term = self.mem_pool.sample(batch_size=self.args.rl_bs)
                q_sa_list, q_target_list, q_real_list = [], [], []
                for i in range(self.args.rl_bs):
                    # [candidate, 3], [bs], [1], (next_state, mask), bool
                    s, a, r, s_prime, is_terminal = list_states[i], list_actions[i], list_rewards[i], list_s_primes[i], list_term[i]
                     
                    predicted_q_value = self.net(s, self.G.adj_list)
                    q_sa = torch.mean(predicted_q_value[a, :]).view(-1)
                    q_target = r.cuda()
                    
                    q_real_list.append(r)
 
                    if not is_terminal:
                        q_t_plus = self.old_net(s_prime[0], self.G.adj_list).detach()
                        q_t_plus_policy = self.net(s_prime[0], self.G.adj_list).detach()

                        in_pool = torch.where(s_prime[1]==0)[0]
                        in_pool_policy_value = q_t_plus_policy[in_pool, :].view(-1)

                        values, indices = torch.topk(in_pool_policy_value, k=self.args.query_bs)

                        in_pool_value = q_t_plus[in_pool, :]
                        q_maxi_t_plus = torch.mean(in_pool_value[indices, :])
                        q_target += self.args.gamma * q_maxi_t_plus
                    
                    q_sa_list.append(q_sa)
                    q_target_list.append(q_target)

                q_sa = torch.stack(q_sa_list)
                q_target = torch.stack(q_target_list)
                q_real = torch.stack(q_real_list)

                loss = F.smooth_l1_loss(q_sa, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (self.step+1) % 20 == 0:
                print("Time:{:.2f}\tIteration:{}\tEps:{:.5f}\tLoss:{:.5f}\tQ_real:{:.5f}\tQ_target:{:.5f}".format(time.time()-tf, self.step, self.eps, loss.item(), torch.mean(q_real).item(), torch.mean(q_target).item()))
                tf = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

if __name__ == '__main__':
    args = parse_args()
    model_name = 'sumdqn_s4_{}'.format(args.dataset)
    args.query_bs = 15
    args.query_cs = 5
    
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    log_file = "./logs/{}.txt".format(model_name)
    Tee(log_file, "w")
    model_save_dir = './models/{}'.format(model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    print_rl_params(args)
    
    G = GraphLoader(args.dataset)
    tgt_G = GraphLoader(args.dataset)
    
    p = Player(G, args)
    tgt_p = Player(G, args)

    env = Env(p, tgt_p, args)
    agent = Agent(args, env, G, tgt_G, model_save_dir)
    
    # tf = time.time()
    # agent.run_simulation()
    # agent.save_transitions()
    # print("Time takes:{:.2f}".format(time.time()-tf))
    
    # tf = time.time()
    # agent.eval()
    # print("Time takes:{:.2f}".format(time.time()-tf))
    
    agent.train()



    

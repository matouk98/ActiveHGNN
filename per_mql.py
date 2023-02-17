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
from mem import NstepReplayMemCell, MQLMem
from utils.config import parse_args
from utils.utils import Tee, greedy_actions, uniform_actions, print_res, print_mql_params, Euclidean_Distance, get_graph

def percd(input):
    ranks = torch.argsort(input)
    ranks = (torch.argsort(ranks) + 1.0) / input.shape[0]
    return ranks

def load_transition(args):
    info = torch.load(tran_path)
    actions = info['action']
    rewards = info['reward']
    states = info['state']
    sprimes = info['sprime']
    masks = info['masks']
    transitions = []
    for i in range(args.query_cs):
        transitions.append([])

    init_num = args.ntest + args.nval + args.query_init
    n = actions.shape[0]
    for i in range(n):
        action = actions[i]
        reward = rewards[i]
        state = states[i]
        if torch.sum(sprimes[i]).item() == 0:
            terminal = True
            sprime = None
            now_step = int(args.query_cs)
        else:
            terminal = False
            sprime = (sprimes[i], masks[i])
            now_step = int((torch.sum(masks[i]).item() - init_num) / args.query_bs)

        transition = (state, action, reward, sprime, terminal)
        transitions[now_step-1].append(transition)

    return transitions

def make_actions(net, env, G, pool, greedy=True): 
    adj_list = G.adj_list    
    cur_state = env.get_state(dataset='target').detach()
    values = net(cur_state, adj_list).detach()

    actions = greedy_actions(pool, values, args.query_bs)
    actions = torch.LongTensor(actions).view(-1)
    q_values = values[actions]
        
    return cur_state, actions, q_values

def train(args, net, old_net, G, tgt_G, mem_pool):
    optimizer = optim.Adam(net.parameters(), lr=args.rllr)
    tf = time.time()

    for step in range(args.mql_epoch):        
        for t in range(args.rl_train_iters):
            list_states, list_actions, list_rewards, list_s_primes, list_term, list_score = mem_pool.sample(batch_size=args.rl_bs)
            scores = torch.FloatTensor(list_score).cuda()
            q_sa_list, q_target_list, q_real_list = [], [], []
            for i in range(args.rl_bs):
                # [candidate, 3], [bs], [1], (next_state, mask), bool
                s, a, r, s_prime, is_terminal = list_states[i], list_actions[i], list_rewards[i], list_s_primes[i], list_term[i]
                     
                predicted_q_value = net(s.cuda(), G.adj_list)
                q_sa = torch.mean(predicted_q_value[a, :]).view(-1)
                q_target = r.cuda()
                    
                q_real_list.append(r)
 
                if not is_terminal:
                    q_t_plus = old_net(s_prime[0].cuda(), G.adj_list).detach()
                    q_t_plus_policy = net(s_prime[0].cuda(), G.adj_list).detach()

                    in_pool = torch.where(s_prime[1]==0)[0]
                    in_pool_policy_value = q_t_plus_policy[in_pool, :].view(-1)

                    values, indices = torch.topk(in_pool_policy_value, k=args.query_bs)

                    in_pool_value = q_t_plus[in_pool, :]
                    q_maxi_t_plus = torch.mean(in_pool_value[indices, :])
                    q_target += args.gamma * q_maxi_t_plus
                    
                q_sa_list.append(q_sa)
                q_target_list.append(q_target)

            q_sa = torch.stack(q_sa_list)
            q_target = torch.stack(q_target_list)
            q_real = torch.stack(q_real_list)

            loss = F.smooth_l1_loss(q_sa, q_target, reduction='none')
            if score_flag:
                loss = torch.mean(loss * scores)
            else:
                loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (step+1) % 20 == 0:
        #     print("Time:{:.2f}\tIteration:{}\tLoss:{:.5f}\tQ_real:{:.5f}\tQ_target:{:.5f}".format(time.time()-tf, step+1, loss.item(), torch.mean(q_real).item(), torch.mean(q_target).item()))
        #     tf = time.time()

        if (step+1) % args.mql_update == 0:
            old_net.load_state_dict(net.state_dict())

def add_to_mem_with_score(mem_pool, transitions, score):
    for i, (s, a, r, s_prime, flag) in enumerate(transitions):
        if score[i] > score_threshold:
            mem_pool.add(s, a, r, s_prime, flag, score[i])

def calc_embed_match(transitions, net, tgt_state, pool, G, tgt_G):
    from scipy.optimize import linear_sum_assignment
    n = len(transitions)
    n_action = transitions[0][1].shape[0]
    assert n_action == args.query_bs
    embeds = []
    tgt_embed = net.get_embed(tgt_state.cuda(), tgt_G.adj_list).detach().cpu()
    tgt_values = net(tgt_state.cuda(), tgt_G.adj_list).detach().cpu()
    tgt_actions = greedy_actions(pool, tgt_values, n_action)
    tgt_action_embed = tgt_embed[tgt_actions, :]
    costs = []

    for s, a, r, s_prime, flag in transitions:
        embed = net.get_embed(s.cuda(), G.adj_list).detach().cpu()
        action_embed = embed[a, :]
        dist = Euclidean_Distance(action_embed, tgt_action_embed)
        row, col = linear_sum_assignment(dist)
        cost = dist[row, col].sum()
        costs.append(cost)

    costs = torch.FloatTensor(costs)
    score = percd(-costs)
    return score

def build_mem(transitions, net, tgt_state, pool, G, tgt_G):
    if similarity_method == 'embed_match':
        score = calc_embed_match(transitions, net, tgt_state, pool, G, tgt_G)
    mem_pool = MQLMem(memory_size=5000)
    add_to_mem_with_score(mem_pool, transitions, score)

    return mem_pool

def permql(args, env, net, old_net, G, tgt_G, src_transitions):
    be = time.time()
    env.reset(dataset='target')
    env.init_train(dataset='target')
    accs, macro_f1s, micro_f1s = [], [], []
    pre_net = QNet(args, env.statedim).cuda()
    pre_net.load_state_dict(old_net.state_dict())
    
    for i in range(args.query_cs):
        if torch.sum(env.tgt_player.trainmask).item() != i * args.query_bs + args.query_init:
            print("error!")
            break
        
        state_t = env.get_state(dataset='target').detach()
        pool = env.tgt_player.get_pool()
        mem_pool = build_mem(src_transitions[i], old_net, state_t, pool, G, tgt_G)
        
        train(args, net, old_net, G, tgt_G, mem_pool)
        state_t, action_t, value_t = make_actions(net, env, tgt_G, pool, greedy=True)

        acc, macro_f1, micro_f1 = 0.0, 0.0, 0.0
        for j in range(eval_times):
            env.tgt_player.model_reset()
            if j == 0:
                query_flag = True
            else:
                query_flag = False
            acc_h, macro_f1_h, micro_f1_h = env.step(action_t, dataset='target', query=query_flag)
            acc += acc_h
            macro_f1 += macro_f1_h
            micro_f1 += micro_f1_h
        
        acc /= eval_times
        macro_f1 /= eval_times
        micro_f1 /= eval_times

        accs.append(acc)
        macro_f1s.append(macro_f1)
        micro_f1s.append(micro_f1)
        old_net.load_state_dict(pre_net.state_dict())
        net.load_state_dict(pre_net.state_dict())

    accs, macro_f1s, micro_f1s = np.array(accs), np.array(macro_f1s), np.array(micro_f1s)
    print_res(accs, macro_f1s, micro_f1s)
    print("Time takes:{:.2f}".format(time.time() - be))
    return accs, macro_f1s, micro_f1s

def test(args, env, old_net, tgt_G):
    be = time.time()
    env.reset(dataset='target')
    env.init_train(dataset='target')
    accs, macro_f1s, micro_f1s = [], [], []
    
    for i in range(args.query_cs):
        if torch.sum(env.tgt_player.trainmask).item() != i * args.query_bs + args.query_init:
            print("error!")
            break
        
        state_t = env.get_state(dataset='target').detach()
        pool = env.tgt_player.get_pool()
        state_t, action_t, value_t = make_actions(old_net, env, tgt_G, pool, greedy=True)

        acc, macro_f1, micro_f1 = 0.0, 0.0, 0.0
        for j in range(eval_times):
            env.tgt_player.model_reset()
            if j == 0:
                query_flag = True
            else:
                query_flag = False
            acc_h, macro_f1_h, micro_f1_h = env.step(action_t, dataset='target', query=query_flag)
            acc += acc_h
            macro_f1 += macro_f1_h
            micro_f1 += micro_f1_h
        
        acc /= eval_times
        macro_f1 /= eval_times
        micro_f1 /= eval_times

        accs.append(acc)
        macro_f1s.append(macro_f1)
        micro_f1s.append(micro_f1)
        
    accs, macro_f1s, micro_f1s = np.array(accs), np.array(macro_f1s), np.array(micro_f1s)
    return accs, macro_f1s, micro_f1s

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
model_v = 'sumdqn_s4_acm'
src_dataset = model_v.split('_')[-3]
model_path = './models/{}.model'.format(model_v)
tran_path = './models/{}_tran.pt'.format(model_v)
repeat_times = 20
eval_times = 5
score_flag = True
score_threshold = 0.5
similarity_method = 'embed_match'

if __name__ == '__main__':
    args = parse_args()
    args.dataset = model_v.split('_')[-1]
    args.tgt_dataset = 'dblp'
    args.rllr = 0.0005
    args.mql_epoch = 20
    args.query_bs = 15
    args.query_cs = 5

    model_name = '{}_{}'.format(args.dataset, args.tgt_dataset)

    os.makedirs("./logs", exist_ok=True)
    log_file = "./logs/{}.txt".format(model_name)
    Tee(log_file, "w")

    print(model_v)
    
    print_mql_params(args)
    print("Repeat Times:{}".format(repeat_times))
    print("Score Flag:{}".format(score_flag))
    print("Score Threshold:{:.2f}".format(score_threshold))
    
    src_transitions = load_transition(args)
    
    G = get_graph(args.dataset)
    tgt_G = get_graph(args.tgt_dataset)
    
    p = Player(G, args)
    tgt_p = Player(tgt_G, args)
    env = Env(p, tgt_p, args)

    net = QNet(args, env.statedim).cuda()
    old_net = QNet(args, env.statedim).cuda()
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt)
    old_net.load_state_dict(ckpt)
    old_net.eval()

    tf = time.time()
    avg_acc, avg_macro, avg_micro = np.zeros(args.query_cs), np.zeros(args.query_cs), np.zeros(args.query_cs)
    
    for i in range(repeat_times):
        accs, macro_f1s, micro_f1s = permql(args, env, net, old_net, G, tgt_G, src_transitions)
        avg_acc += accs
        avg_macro += macro_f1s
        avg_micro += micro_f1s

    avg_acc /= repeat_times
    avg_macro /= repeat_times
    avg_micro /= repeat_times
    print_res(avg_acc, avg_macro, avg_micro)
    print("Time takes:{:.2f}".format(time.time()-tf))

    # tf = time.time()
    # old_net = QNet(args, env.statedim).cuda()
    # old_net.load_state_dict(torch.load(model_path))
    # old_net.eval()
    # avg_acc, avg_macro, avg_micro = np.zeros(args.query_cs), np.zeros(args.query_cs), np.zeros(args.query_cs)
    # p = Player(G, args)
    # tgt_p = Player(tgt_G, args)
    # env = Env(p, tgt_p, args)

    # for i in range(repeat_times):
    #     accs, macro_f1s, micro_f1s = test(args, env, old_net, tgt_G)
    #     avg_acc += accs
    #     avg_macro += macro_f1s
    #     avg_micro += micro_f1s

    # avg_acc /= repeat_times
    # avg_macro /= repeat_times
    # avg_micro /= repeat_times
    # print_res(avg_acc, avg_macro, avg_micro)
    # print("Time takes:{:.2f}".format(time.time()-tf))
    
    


    

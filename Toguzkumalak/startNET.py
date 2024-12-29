import math, random, inspect, time, random, os, psutil, collections, sys
import numpy as np
from timeit import default_timer as timer
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import pickle

from mcts import *

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = [item[0] for item in dataset]
        self.y_p, self.y_v = [item[1] for item in dataset], [item[2] for item in dataset]
#         self.X = dataset[:,0]
#         self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].transpose(0, 1), self.y_p[idx], self.y_v[idx]

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
def net_func(game):
    cuda = torch.cuda.is_available()
    encoded_s = None
    if cuda:
        encoded_s = torch.from_numpy(encodeBoard(game)).cuda()
    else:
        encoded_s = torch.from_numpy(encodeBoard(game))
    child_priors, value_estimate = net(encoded_s)
    return child_priors.detach().numpy()[0], value_estimate.detach().numpy()[0]

def save_as_pickle(filename, data):
    completeName = os.path.join("/kaggle/working/datasets/iter2/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class Player:
    def __init__(self, name="Unknown player"):
        self.score = 0
        self.name = name
        
class Human(Player):
    def __init__(self, name):
        Player.__init__(self, name)
        
class AI(Player):
    def __init__(self, name):
        Player.__init__(self, name)
    def getMove(self, game):
        pass

class RandomAI(AI):
    def __init__(self, name):
        AI.__init__(self, name)
    def getMove(self, game):
        return random.choice(game.getPossibleMoves())
       
class InputLayers(nn.Module):
    def __init__(self, action_size):
        super(InputLayers, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(3, 162, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(162)

    def forward(self, x):
        x = x.view(-1, 3, 9, 2)  # batch_size x channels x board_x x board_y
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class BoardFeatures(nn.Module):
    def __init__(self, inplanes=162, planes=162):
        super(BoardFeatures, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutputLayers(nn.Module):
    def __init__(self, action_size):
        super(OutputLayers, self).__init__()
        self.conv_value = nn.Conv2d(162, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc1_value = nn.Linear(2 * 9, 18)
        self.fc2_value = nn.Linear(18, 1)

        self.conv_policy = nn.Conv2d(162, 81, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(81)
        self.fc_policy = nn.Linear(2 * 9 * 81, action_size)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, board_features):
        value_head = F.relu(self.bn_value(self.conv_value(board_features)))
        value_head = value_head.view(-1, 2 * 9)
        value_head = F.relu(self.fc1_value(value_head))
        value_output = torch.tanh(self.fc2_value(value_head))

        policy_head = F.relu(self.bn_policy(self.conv_policy(board_features)))
        policy_head = policy_head.view(-1, 2 * 9 * 81)
        policy_output = self.fc_policy(policy_head)
        policy_output = self.logsoftmax(policy_output).exp()

        return policy_output, value_output

class TNET(nn.Module):
    def __init__(self):
        super(TNET, self).__init__()
        self.conv = InputLayers(action_size=9)
        for block in range(19):
            setattr(self, "res_%i" % block,BoardFeatures())
        self.outblock = OutputLayers(action_size=9)
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
    
    def act(self, state, game):
        act_values, policy = None, None
        cuda = torch.cuda.is_available()
        if cuda:
            act_values, policy = self(state.cuda())
        else:
            act_values, policy = self(state)
        #print(act_values)
        possible_moves = game.getPossibleMoves()
        for i in range(9):
            if (i, game.player) not in possible_moves:
                act_values[0][i] = np.NINF
        #print(act_values)
        if cuda:
            return (np.argmax(act_values[0].cpu().detach().numpy()), game.player)  # лучшее действие
        return (np.argmax(act_values[0].detach().numpy()), game.player)  # лучшее действие
        
# def encodeBoard(game: Game):
#     tuzdyk1 = game.tuzdyk1
#     tuzdyk2 = game.tuzdyk2
#     player = game.player
#     input_board = game.copyBoard(game.board)
#     kazan = np.array([game.player1_score, game.player2_score])
#     for i, row in enumerate(input_board):
#         for j, cell in enumerate(input_board[i]):
#             playerSide = 0
#             if player == i:
#                 playerSide = 1
#             input_board[i][j] = [game.board[i][j], 0, playerSide]
#             if i == 0 and tuzdyk2 == j:
#                 input_board[i][j] = [game.board[i][j], 1, playerSide]
#             if i == 1 and tuzdyk1 == j:
#                 input_board[i][j] = [game.board[i][j], 1, playerSide]
#     input_board = np.array(input_board)#.reshape(1, 2, 9, 3)
#     torch_tensor = torch.from_numpy(input_board).float()
#     return torch_tensor
        
def self_play(net, episodes_per_process = 25):
    tnet = net
    #tnet.load("/kaggle/working/togyzkumalak_333.keras")
    
    randomPlayer = RandomAI("Random")
    game = Game()
    white_wins=0
    black_wins=0
    # Параметры обучения
    n_episodes = episodes_per_process
    dataset = []  # Хранение данных для обучения
    #data = load_dataset_from_pickle("togyzkumalak_dataset.pkl")
    for episode in range(n_episodes):
        game.reset()
        episode_data = []  # Данные для текущего эпизода
        value = 0
        while True:
            #game.showBoard()
            action = None
            if game.checkWinner() != -1:
                if game.checkWinner() == 1: # black wins
                    value = -1
                    black_wins+=1
                elif game.checkWinner() == 0: # white wins
                    value = 1
                    white_wins+=1
                print(f"Episode {episode + 1}/{n_episodes} Score: {game.player1_score}-{game.player2_score} Moves: {game.fullTurns}, Results: {white_wins}-{black_wins}")
                break
            if game.player == 0:
                state = encodeBoard(game)
                action = tnet.act(state, game)[0]
            else:
                action = randomPlayer.getMove(game)[0]
            #move = (action, 0 if board.turnOwner == board.player1 else 1)
            game.makeMove(action, game.player)
    return tnet, dataset

def train(net, dataset, epoch_start=0, epoch_stop=20, cpu=0):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
    batch_size=30
    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        total_loss = 0.0
        losses_per_batch = []
        for i,data in enumerate(train_loader,0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.float().cuda(), policy.float().cuda(), value.float().cuda()
            optimizer.zero_grad()
            policy_pred, value_pred = net(state) # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1)*batch_size, len(train_set), total_loss/10))
                print("Policy:",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Value:",value[0].item(),value_pred[0,0].item())
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.01:
                break

def MCTS_self_play(net = None, num_games = 25, cpu = 0):
    start_time = datetime.now()
    curr_time = start_time.strftime("%H:%M:%S")
    print(f"[{curr_time}] Process {cpu} started")
    for idxx in range(0, num_games):
        game = Game()
        dataset = []  # Данные для текущего эпизода
        value = 0
        while True:
            print(str(datetime.now() - start_time).split(".")[0])
            game.showBoard()
            if game.checkWinner() != -1 or game.semiTurns > 100:
                curr_time = datetime.now().strftime("%H:%M:%S")
                if game.checkWinner() == 1: # black wins
                    value = -1
                elif game.checkWinner() == 0: # white wins
                    value = 1
                print(f"[{curr_time}] [{cpu}]-Episode {idxx + 1}/{num_games} Score: {game.player1_score}-{game.player2_score} Turns: {game.semiTurns}")
                break
            
            root_action, root = UCT_search(game, 800, net_func)
            print(root_action)
            state = encodeBoard(game)
            policy = get_policy(root)
            print(policy)
            dataset.append((state, policy))
            game.makeMove(root_action, game.player)
        dataset_p = []
        for idx, data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset
        filename = "dataset_cpu%i_%i_%s.pkl" % (cpu, idxx, datetime.today().strftime("%Y-%m-%d"))
        save_as_pickle(filename, dataset_p)
    end_time = datetime.now()
    curr_time = end_time.strftime("%H:%M:%S")
    elapsed = str(end_time - start_time).split(".")[0]
    print(f"[{curr_time}] Process {cpu} completed. Elapsed {elapsed}.")
    return tnet

def get_batched_dataset(path = '/kaggle/working/combined_dataset.pkl'):
    # Загрузка объединенного датасета
    dataset_file = path
    with open(dataset_file, 'rb') as f:
        combined_data = pickle.load(f)
    print("__________________________")
    print(len(combined_data))
    print("__________________________")
    # Перемешивание данных
    #combined_data = shuffle(combined_data)
    return combined_data
    
def start_train(tnet):
    data = get_batched_dataset("/kaggle/working/datasets/combined/combined_dataset.pkl")
    print(data[0])
    train(tnet, data, 0, 200)
    torch.save({'state_dict': tnet.state_dict()}, os.path.join("./model_data/",\
                                "current_trained_net.pth.tar"))
    print("Saved")
    
from joblib import Parallel, delayed
if __name__ == "__main__":
    cpus = mp.cpu_count()
    num_processes = torch.cuda.device_count()
    episodes_per_process = 50
    print("Device count:", num_processes)
    print("CPU count:", cpus)
    net_to_play="current_trained_net.pth.tar"
    #mp.set_start_method("spawn",force=True)
    net = TNET()
    cuda = torch.cuda.is_available()
    print("CUDA found:", cuda)
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                  "current_net.pth.tar"))
    
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_play)
    #device = torch.device('cuda' if cuda else 'cpu')
    #checkpoint = torch.load(current_net_filename, map_location=device)
    #net.load_state_dict(checkpoint['state_dict'])
    #self_play(net, 10)
    MCTS_self_play(net, 50, 0)
    #for i in range(1):
    #    start_train(net)
    # Установите количество желаемых процессов
    #processes = []
    #for i in range(cpus):
    #    p = mp.Process(target=MCTS_self_play, args=(net, episodes_per_process, i))
    #    p.start()
    #    processes.append(p)
    #for p in processes:
    #    p.join()
    #Parallel(n_jobs=80)(delayed(MCTS_self_play)(net, episodes_per_process, i) for i in range(cpus))
    self_play(net, 10)
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt

# NodeType: INPUT, HIDDEN, or OUTPUT.
class NodeType:
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2

# MemoryCellType: enumerates supported hidden cell types.
class MemoryCellType:
    DELTA = "delta"   # custom
    GRU   = "gru"     # uses nn.GRUCell
    LSTM  = "lstm"    # uses nn.LSTMCell
    MGU   = "mgu"     # custom
    UGRNN = "ugrnn"   # custom
    SIMPLE= "simple"  # uses nn.RNNCell

# Represents a node (ID, type, cell type).
class NodeGene:
    def __init__(self, node_id, node_type, cell_type):
        self.node_id = node_id
        self.node_type = node_type
        self.cell_type = cell_type

# Represents an edge (in->out), possibly recurrent with time_skip>1, and can be enabled/disabled.
class EdgeGene:
    def __init__(self, in_id, out_id, is_recurrent=False, time_skip=1, enabled=True):
        self.in_id = in_id
        self.out_id = out_id
        self.is_recurrent = is_recurrent
        self.time_skip = time_skip
        self.enabled = enabled

class DeltaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, x, h):
        delta = torch.tanh(self.x2h(x) + self.h2h(h) + self.bias)
        return h + delta

class MGUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.xf = nn.Linear(input_size, hidden_size, bias=False)
        self.hf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.xn = nn.Linear(input_size, hidden_size, bias=False)
        self.hn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, x, h):
        f = torch.sigmoid(self.xf(x) + self.hf(h) + self.bf)
        n = torch.tanh(self.xn(x) + self.hn(h) + self.bn)
        return f * h + (1 - f) * n

class UGRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.xz = nn.Linear(input_size, hidden_size, bias=False)
        self.hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bz = nn.Parameter(torch.zeros(hidden_size))
        self.xh = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, x, h):
        z = torch.sigmoid(self.xz(x) + self.hz(h) + self.bz)
        h_tilde = torch.tanh(self.xh(x) + self.hh(h) + self.bh)
        return (1 - z) * h + z * h_tilde

# Utility module: extracts final time-step output from (B, T, Hidden).
class ExtractLastTimeStep(nn.Module):
    def forward(self, x):
        if isinstance(x, tuple):
            outputs, hidden = x
            return outputs[:, -1, :]
        return x[:, -1, :]

# If no hidden nodes, do a minimal seed net: average over time dimension -> linear layer.
class MinimalSeedNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        avg = x.mean(dim=1)
        return self.linear(avg)

class MixedRNNBlock(nn.Module):
    def __init__(self, input_size, node_cells):
        super().__init__()
        self.node_cells = nn.ModuleList(node_cells)
        self.input_size = input_size
    def forward(self, x):
        b, t, _ = x.shape
        h_states = [None]*len(self.node_cells)
        c_states = [None]*len(self.node_cells)
        outputs  = []
        for step in range(t):
            xt = x[:, step, :]
            step_outs = []
            for idx, cell in enumerate(self.node_cells):
                if isinstance(cell, nn.LSTMCell):
                    if h_states[idx] is None:
                        h_states[idx] = torch.zeros(b, cell.hidden_size, device=xt.device)
                        c_states[idx] = torch.zeros(b, cell.hidden_size, device=xt.device)
                    h_states[idx], c_states[idx] = cell(xt, (h_states[idx], c_states[idx]))
                    step_outs.append(h_states[idx])
                else:
                    if h_states[idx] is None:
                        h_states[idx] = torch.zeros(b, cell.hidden_size, device=xt.device)
                    h_states[idx] = cell(xt, h_states[idx])
                    step_outs.append(h_states[idx])
            cat = torch.cat(step_outs, dim=1)
            outputs.append(cat.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# Genome: holds a list of NodeGene/EdgeGene, builds the corresponding PyTorch model.
class Genome:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.model = None

    # clone() includes weight copying with strict=False to allow structural changes
    def clone(self):
        g = Genome()
        g.nodes = copy.deepcopy(self.nodes)
        g.edges = copy.deepcopy(self.edges)
        if self.model is not None:
            g.build_model()
            g.model.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=False)
        return g

    # build_model: constructs either a minimal seed or a MixedRNNBlock + linear layer
    def build_model(self):
        in_count = sum(n.node_type == NodeType.INPUT for n in self.nodes)
        out_count= sum(n.node_type == NodeType.OUTPUT for n in self.nodes)
        hidden_nodes = [n for n in self.nodes if n.node_type==NodeType.HIDDEN]
        if len(hidden_nodes) == 0:
            self.model = MinimalSeedNetwork(in_count, out_count)
            return
        cell_list = []
        for h in hidden_nodes:
            if h.cell_type==MemoryCellType.LSTM:
                cell_list.append(nn.LSTMCell(in_count, 1))
            elif h.cell_type==MemoryCellType.GRU:
                cell_list.append(nn.GRUCell(in_count, 1))
            elif h.cell_type==MemoryCellType.SIMPLE:
                cell_list.append(nn.RNNCell(in_count, 1))
            elif h.cell_type==MemoryCellType.MGU:
                cell_list.append(MGUCell(in_count, 1))
            elif h.cell_type==MemoryCellType.UGRNN:
                cell_list.append(UGRNNCell(in_count, 1))
            elif h.cell_type==MemoryCellType.DELTA:
                cell_list.append(DeltaRNNCell(in_count, 1))
            else:
                cell_list.append(DeltaRNNCell(in_count, 1))
        rnn_block = MixedRNNBlock(in_count, cell_list)
        fc_out    = nn.Linear(len(hidden_nodes), out_count)
        self.model= nn.Sequential(rnn_block, ExtractLastTimeStep(), fc_out)

    # mutate_add_node: splits one enabled edge, inserting a new hidden node between them
    def mutate_add_node(self):
        en = [e for e in self.edges if e.enabled]
        if not en:
            return
        picked = random.choice(en)
        picked.enabled = False
        new_id = max(n.node_id for n in self.nodes)+1
        pick_cell = random.choice([
            MemoryCellType.DELTA,
            MemoryCellType.GRU,
            MemoryCellType.LSTM,
            MemoryCellType.MGU,
            MemoryCellType.UGRNN,
            MemoryCellType.SIMPLE
        ])
        new_node = NodeGene(new_id, NodeType.HIDDEN, pick_cell)
        self.nodes.append(new_node)
        e1 = EdgeGene(picked.in_id, new_id, picked.is_recurrent, picked.time_skip, True)
        e2 = EdgeGene(new_id, picked.out_id, False, 1, True)
        self.edges.extend([e1,e2])

    # mutate_add_connection: adds a new edge between valid in->out nodes, possibly recurrent
    def mutate_add_connection(self):
        possible_in = [n.node_id for n in self.nodes if n.node_type!=NodeType.OUTPUT]
        possible_out= [n.node_id for n in self.nodes if n.node_type!=NodeType.INPUT]
        for _ in range(10):
            if not possible_in or not possible_out:
                return
            src = random.choice(possible_in)
            dst = random.choice(possible_out)
            if src!=dst and not any(e.in_id==src and e.out_id==dst for e in self.edges):
                if random.random()<0.3:
                    rec=True
                    skip= random.randint(1,10)
                else:
                    rec=False
                    skip=1
                self.edges.append(EdgeGene(src, dst, rec, skip, True))
                return

    # mutate_perturb_weights: adds small random noise to existing parameters
    def mutate_perturb_weights(self):
        if self.model is None:
            return
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(0.01 * torch.randn_like(p))

    # mutate: chooses from one of the three operations
    def mutate(self):
        ops = [self.mutate_add_node, self.mutate_add_connection, self.mutate_perturb_weights]
        random.choice(ops)()

    @staticmethod
    def crossover(p1, p2):
        child = p1.clone()
        p2map = {(e.in_id, e.out_id): e for e in p2.edges if e.enabled}
        updated=[]
        for e in child.edges:
            if (e.in_id, e.out_id) in p2map and random.random()<0.5:
                ref = p2map[(e.in_id, e.out_id)]
                e.enabled = ref.enabled
                e.is_recurrent = ref.is_recurrent
                e.time_skip = ref.time_skip
            updated.append(e)
        child.edges = updated
        for k, e2 in p2map.items():
            if not any(ec.in_id==e2.in_id and ec.out_id==e2.out_id for ec in child.edges):
                if random.random()<0.3:
                    child.edges.append(copy.deepcopy(e2))
        c_ids= [n.node_id for n in child.nodes]
        for n2 in p2.nodes:
            if n2.node_id not in c_ids and random.random()<0.3:
                child.nodes.append(copy.deepcopy(n2))
        return child

# train_genome: trains a genome on 600 (max) random subsequences for a certain # of epochs
def train_genome(genome, subsequences, batch_size=32, noise_std=0.05, epochs=20):
    if not subsequences:
        return
    count = min(len(subsequences), 600)
    idxs  = random.sample(range(len(subsequences)), count)
    sample_data = [subsequences[i] for i in idxs]
    genome.build_model()
    model = genome.model
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    def batcher(arr):
        random.shuffle(arr)
        for i in range(0, len(arr), batch_size):
            block = arr[i:i+batch_size]
            bx    = np.stack(block)
            Xb    = torch.tensor(bx, dtype=torch.float).unsqueeze(-1).to(device)
            Yb    = torch.tensor(bx[:, -1], dtype=torch.float).unsqueeze(-1).to(device)
            yield Xb, Yb

    for ep in range(epochs):
        add_noise = (ep >= epochs//2)
        for Xb, Yb in batcher(sample_data):
            if add_noise and noise_std>0:
                stdev = torch.std(Xb)
                noise = noise_std * stdev * torch.randn_like(Xb)
                Xb    = Xb + noise
            opt.zero_grad()
            out  = model(Xb)
            loss = crit(out, Yb)
            loss.backward()
            opt.step()

# evaluate_genome: checks MSE on last 100 subsequences
def evaluate_genome(genome, subsequences):
    if not subsequences:
        return float('inf')
    block_size= min(len(subsequences), 100)
    block     = subsequences[-block_size:]
    genome.build_model()
    model = genome.model
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    crit = nn.MSELoss()
    total=0
    with torch.no_grad():
        for seq in block:
            Xb= torch.tensor(seq, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)
            Yb= torch.tensor(seq[-1], dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)
            pred= model(Xb)
            total+= crit(pred, Yb).item()
    return total/block_size

# Island: holds an elite set and a newly generated population
class Island:
    def __init__(self, island_id, elite_size):
        self.island_id = island_id
        self.elite_size= elite_size
        self.elite      = []
        self.population = []
    def get_all_genomes(self):
        return self.elite + self.population
    def update_elite(self, new_elite):
        self.elite = new_elite

def load_djia_subsequences(file_path, seq_length=25):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True, ignore_index=True)
    arr = df['DJIA'].values
    subs= []
    for i in range(len(arr)-seq_length+1):
        subs.append(arr[i:i+seq_length])
    return np.array(subs), arr

def arima_forecast(train_data, steps_ahead=1, order=(5,1,0)):
    model= sm.tsa.ARIMA(train_data, order=order)
    fitted= model.fit()
    return fitted.forecast(steps=steps_ahead)

class OneNAS:
    def __init__(self,
                 file_path="./datasets/DJA.csv",
                 seq_length=25,
                 generations=50,
                 island_count=2,
                 elite_size=2,
                 island_pop_size=3,
                 extinct_freq=10,
                 mutation_rate=0.4,
                 crossover_rate=0.6):
        self.file_path   = file_path
        self.seq_length  = seq_length
        self.generations = generations
        self.island_count= island_count
        self.elite_size  = elite_size
        self.island_pop_size= island_pop_size
        self.extinct_freq= extinct_freq
        self.mutation_rate= mutation_rate
        self.crossover_rate= crossover_rate

        self.data_subseqs, self.full_series = load_djia_subsequences(self.file_path, self.seq_length)
        if self.generations> len(self.data_subseqs):
            raise ValueError("Not enough subsequences for the requested generations.")

        self.islands             = []
        self.global_best_genome  = None
        self.global_best_mse     = float('inf')
        self.historical_data     = []
        self.predictions_over_time= []
        self.actual_over_time    = []
        self.mse_history         = []

        # Set up a live figure with 2 subplots: one for the walk-forward time series, one for MSE
        plt.ion()
        self.fig, (self.ax_ts, self.ax_mse) = plt.subplots(2,1, figsize=(10,8))
        self.fig.tight_layout(pad=3.0)
        self.ax_ts.set_title("Walk-Forward Time Series (Partial)")
        self.ax_ts.set_xlabel("Generation")
        self.ax_ts.set_ylabel("Value")
        self.ax_mse.set_title("Global Best MSE Over Generations (Partial)")
        self.ax_mse.set_xlabel("Generation")
        self.ax_mse.set_ylabel("MSE")
        plt.show(block=False)

    def initialize_islands(self):
        for i in range(self.island_count):
            isl = Island(i, self.elite_size)
            self.islands.append(isl)

        # minimal seed with 1 input node, 1 output node
        seed = Genome()
        seed.nodes = [
            NodeGene(0, NodeType.INPUT, MemoryCellType.SIMPLE),
            NodeGene(1, NodeType.OUTPUT,MemoryCellType.SIMPLE)
        ]
        seed.edges= [EdgeGene(0, 1, False, 1, True)]
        seed.build_model()

        self.global_best_genome= seed.clone()
        for isl in self.islands:
            isl.elite= [seed.clone() for _ in range(self.elite_size)]
            isl.population= []

    def extinct_and_repopulate(self):
        scored=[]
        for isl in self.islands:
            top_g = isl.elite[0]
            sc    = evaluate_genome(top_g, self.historical_data)
            scored.append((sc, isl))
        scored.sort(key=lambda x: x[0])
        worst_isl= scored[-1][1]
        worst_isl.elite=[]
        worst_isl.population=[]
        for _ in range(self.elite_size):
            c= self.global_best_genome.clone()
            c.mutate()
            worst_isl.elite.append(c)

    def update_live_plots(self, gidx):
        # Clear current axes
        self.ax_ts.cla()
        self.ax_mse.cla()

        # Plot partial time series (from 0..gidx)
        gens_so_far = list(range(gidx+1))
        partial_preds  = self.predictions_over_time[:gidx+1]
        partial_actual = self.actual_over_time[:gidx+1]
        self.ax_ts.plot(gens_so_far, partial_actual, label='Actual', color='black', linestyle='--')
        self.ax_ts.plot(gens_so_far, partial_preds,  label='Prediction', color='blue')
        self.ax_ts.set_title("Walk-Forward Time Series (0..{})".format(gidx))
        self.ax_ts.set_xlabel("Generation")
        self.ax_ts.set_ylabel("Value")
        self.ax_ts.legend()

        # Plot partial MSE history
        partial_mse = self.mse_history[:gidx+1]
        self.ax_mse.plot(gens_so_far, partial_mse, color='blue', marker='o')
        self.ax_mse.set_title("Global Best MSE Through Gen {}".format(gidx))
        self.ax_mse.set_xlabel("Generation")
        self.ax_mse.set_ylabel("MSE")

        # Make it visually appealing by adjusting axes
        if len(partial_mse) > 0:
            min_m = min(partial_mse)
            max_m = max(partial_mse)
            extra = (max_m - min_m) * 0.1 if max_m>min_m else 1
            self.ax_mse.set_ylim(min_m - extra, max_m + extra)

        self.ax_ts.set_xlim(0, gidx)
        self.ax_mse.set_xlim(0, gidx)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def run(self):
        self.initialize_islands()
        best_for_prediction = self.global_best_genome.clone()

        for gidx in range(self.generations):
            new_sub = self.data_subseqs[gidx]
            self.historical_data.append(new_sub)

            best_for_prediction.build_model()
            pm = best_for_prediction.model
            pm.eval()
            device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pm.to(device)
            with torch.no_grad():
                Xb= torch.tensor(new_sub, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)
                pred_val= pm(Xb).item()

            self.predictions_over_time.append(pred_val)
            self.actual_over_time.append(new_sub[-1])

            if gidx>0 and gidx%self.extinct_freq==0:
                self.extinct_and_repopulate()

            # Generate new population
            for isl in self.islands:
                new_pop=[]
                for _ in range(self.island_pop_size):
                    p1= random.choice(isl.elite)
                    if random.random()< self.crossover_rate:
                        p2= random.choice(isl.elite)
                        child= Genome.crossover(p1,p2)
                    else:
                        child= p1.clone()
                    if random.random()< self.mutation_rate:
                        child.mutate()
                    new_pop.append(child)
                isl.population= new_pop

            # Train each offspring
            for isl in self.islands:
                for c in isl.population:
                    train_genome(c, self.historical_data, epochs=20)

            # Evaluate, update elite
            for isl in self.islands:
                candidates= isl.elite + isl.population
                scored=[]
                for cand in candidates:
                    sc= evaluate_genome(cand, self.historical_data)
                    scored.append((sc,cand))
                scored.sort(key=lambda x: x[0])
                isl.update_elite([x[1] for x in scored[:isl.elite_size]])
                isl.population=[]

            # Update global best
            for isl in self.islands:
                for e in isl.elite:
                    val= evaluate_genome(e, self.historical_data)
                    if val< self.global_best_mse:
                        self.global_best_mse= val
                        self.global_best_genome= e.clone()

            best_for_prediction= self.global_best_genome.clone()
            self.mse_history.append(self.global_best_mse)

            # Every generation, or every N gens, update the live plots
            if gidx%5==0 or gidx==self.generations-1:
                self.update_live_plots(gidx)
            print(f"Generation {gidx} | Current Global Best MSE: {self.global_best_mse:.5f}")

        print("Finished. Final Global Best MSE:", self.global_best_mse)
        self.plot_final()

    def plot_final(self):
        plt.ioff()

        # Final predictions vs actual
        plt.figure(figsize=(10,5))
        plt.plot(self.predictions_over_time, label='ONE-NAS Pred', color='blue')
        plt.plot(self.actual_over_time,       label='Actual',      color='black', linestyle='--')
        plt.title('Final Online Predictions vs Actual (Walk-Forward)')
        plt.xlabel('Generation')
        plt.ylabel('DJIA Value')
        plt.legend()
        plt.show()

        # Compare final ARIMA single-step on last data point if you want
        if len(self.full_series) > self.seq_length:
            idx= len(self.full_series)-1
            trn= self.full_series[:idx]
            act= self.full_series[idx]
            arr= arima_forecast(trn, steps_ahead=1, order=(5,1,0))
            if np.isscalar(arr):
                arr_pred= float(arr)
            else:
                arr_pred= arr[0]
            arr_err= (arr_pred - act)**2
            print("ARIMA single-step MSE on final data point:", arr_err)

if __name__=="__main__":
    budget = OneNAS(
        file_path="./datasets/DJA.csv",
        seq_length=25,
        generations=2000,  # will take years to run
        island_count=2,
        elite_size=2,
        island_pop_size=3,
        extinct_freq=10,
        mutation_rate=0.4,
        crossover_rate=0.6
    )
    budget.run()
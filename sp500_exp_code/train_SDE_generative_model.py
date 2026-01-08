import torch.nn as nn
import torch.optim as optim
import torch
import os
from tqdm import tqdm
torch.set_default_dtype(torch.float32)
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
data_dir = './data/sp500/'
os.makedirs(data_dir, exist_ok=True)


## Load training data
x_dim = 1
sde_T = 1 / 2
sde_dt = 1 / 252
sde_Nt = int(sde_T /sde_dt)
path = torch.load(f'{data_dir}train_sde_path.pt', map_location=device)
path = path.to(dtype=torch.float32)

## Prepare training data: condition and increment
train_condition = path.T.clone()
train_increment = torch.diff(path, dim=1).T.clone()



## Conditional score-based diffusion model and training setup
n_steps = 100
def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas
def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

### Create schedule
betas = make_beta_schedule(schedule='linear', n_timesteps=n_steps, 
                           start=1e-5, end=1e-2).to(device=device)
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

### Define the score network structure
import torch.nn.functional as F
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out
class StepConditionalModel(nn.Module):
    def __init__(self, n_steps, x_dim):
        super(StepConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(x_dim + x_dim, 32, n_steps)
        self.lin2 = ConditionalLinear(32, 32, n_steps)
        self.lin3 = ConditionalLinear(32, 16, n_steps)
        self.lin4 = ConditionalLinear(16, 16, n_steps)
        self.lin5 = nn.Linear(16, x_dim)
    
    def forward(self, state, time, condition):
        x = torch.cat([state, condition], dim=1)
        x = F.softplus(self.lin1(x, time))
        x = F.softplus(self.lin2(x, time))
        x = F.softplus(self.lin3(x, time))
        x = F.softplus(self.lin4(x, time))
        return self.lin5(x)


### Denosing score matching loss
def noise_estimation_loss(model, x_0, condition):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long().to(device)
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0).to(device)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0).to(device)
    e = torch.randn_like(x_0).to(device)
    # model input
    x = x_0 * a + e * am1
    output = model(x, t, condition)
    return (e - output).square().mean()

### DDIM sampler, which is DDPM sampler when eta=1
#### Copied from Equation (12) in https://arxiv.org/abs/2010.02502
@torch.no_grad()
def p_sample_ddim(model_cond, x, t, condition, delta_t = 1, eta=0.1):
    t = torch.tensor([t], device=device).long()
    tp1 = torch.clamp(t - delta_t, 0, n_steps - 1)
    alphas_prod_t = extract(alphas_prod, t, x)
    alphas_prod_tp1 = extract(alphas_prod, tp1, x)
    beta_prod_t = 1 - extract(alphas_prod, t, x)
    beta_prod_tp1 = 1 - extract(alphas_prod, tp1, x)
    sigma2_t = (beta_prod_tp1 / beta_prod_t) * (1 - alphas_prod_t / alphas_prod_tp1)
    std_t = sigma2_t.sqrt() * eta
    pred_eps = model_cond(x, t, condition)
    pred_x0 = (x - beta_prod_t.sqrt() * pred_eps) / alphas_prod_t.sqrt()
    point_dir = (1 - alphas_prod_tp1 - (std_t ** 2)).sqrt() * pred_eps
    sample = alphas_prod_tp1.sqrt() * pred_x0 + point_dir + std_t * torch.randn_like(x)
    return (sample)
@torch.no_grad()
def p_sample_loop_ddim(model_cond, condition, delta_t=1, eta=0.1):
    cur_x = torch.randn_like(condition, device=device)
    for i in reversed(range(0, n_steps, delta_t)):
        cur_x = p_sample_ddim(model_cond, cur_x, i, condition, delta_t, eta)
    return cur_x
@torch.no_grad()
def p_sample_sde_ddim(model_cond, init_condition, delta_t=1, eta=0.1):
    condition = init_condition.clone().view(-1, 1)
    sde_sample_path = [condition.clone()]
    for i in range(sde_Nt):
        condition += p_sample_loop_ddim(model_cond[i], condition, delta_t, eta)
        sde_sample_path.append(condition.clone())
    return sde_sample_path

## Train conditional score-based diffusion model

### Score network and optimizer for each SDE time step
model_cond = [StepConditionalModel(n_steps, x_dim).to(device) for _ in range(sde_Nt)]
optimizer_cond = [optim.Adam(m.parameters(), lr=1e-3) for m in model_cond]

### Training loop
num_iter = 6000
N_sample_path = path.shape[0]
batch_size = N_sample_path // 2
sde_step = list(range(sde_Nt))
for iter in tqdm(range(num_iter)):
    for sde_t in sde_step:
        permutation = torch.randperm(N_sample_path)
        m_cond = model_cond[sde_t]
        opt_cond = optimizer_cond[sde_t]
        for i in range(0, N_sample_path, batch_size):
            opt_cond.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x = train_increment[sde_t][indices].view(batch_size, 1).clone()
            batch_cond = train_condition[sde_t][indices].view(batch_size, 1).clone()
            loss = noise_estimation_loss(m_cond, batch_x, batch_cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_cond.parameters(), 1.)
            opt_cond.step()
    if (iter % (num_iter / 10) == 0):
        print(loss)
    
    ### Learning rate update
    if iter == 3999:
        optimizer_cond = [optim.Adam(m.parameters(), lr=1e-5) for m in model_cond]



## Save the trained conditional score-based diffusion model
torch.save(model_cond, f'{data_dir}ScoreDM_SDE_sp500.pt')



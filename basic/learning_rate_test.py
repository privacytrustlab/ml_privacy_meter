import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from models import get_model
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cyclical_learning_rate(
    batch_step,
    step_size,
    base_lr=0.001,
    max_lr=0.006,
    mode="triangular",
    gamma=0.999995,
):
    cycle = np.floor(1 + batch_step / (2.0 * step_size))
    x = np.abs(batch_step / float(step_size) - 2 * cycle + 1)

    lr_delta = (max_lr - base_lr) * np.maximum(0, (1 - x))

    if mode == "triangular":
        pass
    elif mode == "triangular2":
        lr_delta = lr_delta * 1 / (2.0 ** (cycle - 1))
    elif mode == "exp_range":
        lr_delta = lr_delta * (gamma ** (batch_step))
    else:
        raise ValueError('mode must be "triangular", "triangular2", or "exp_range"')

    lr = base_lr + lr_delta

    return lr


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)
        logits = model(features)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


num_epochs = 25
num_train = 30000
batch_size = 512
base_lr = 0.09
max_lr = 0.5

# Hyperparameters
random_seed = 1
num_classes = 100
np.random.seed(random_seed)
torch.manual_seed(random_seed)


idx = np.arange(50000)  # the size of CIFAR10-train
np.random.shuffle(idx)
val_idx, train_idx = idx[:num_train], idx[num_train:]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_dataset = datasets.CIFAR100(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = datasets.CIFAR100(
    root="data", train=False, transform=transforms.ToTensor()
)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    # shuffle=True, # Subsetsampler already shuffles
    sampler=train_sampler,
)

val_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    # shuffle=True,
    sampler=val_sampler,
)


num_epochs = 10
iter_per_ep = len(train_loader)
model = get_model(model_type="wrn28-2", dataset_name="cifar100")
model = model.to(device)
cost_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=base_lr, nesterov=True, momentum=0.9, weight_decay=5e-4
)
collect = {"lr": [], "cost": [], "train_batch_acc": [], "val_acc": []}

batch_step = -1
cur_lr = base_lr

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
        batch_step += 1
        features = features.to(device)
        targets = targets.to(device)

        ### FORWARD AND BACK PROP
        logits = model(features)
        probas = F.softmax(logits, dim=1)
        cost = cost_fn(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        #############################################
        # Logging
        if not batch_step % 200:
            print(
                "Total batch # %5d/%d" % (batch_step, iter_per_ep * num_epochs), end=""
            )
            print("   Curr. Batch Cost: %.5f" % cost)

        #############################################
        # Collect stats
        model = model.eval()
        train_acc = compute_accuracy(model, [[features, targets]])
        val_acc = compute_accuracy(model, val_loader)
        collect["lr"].append(cur_lr)
        collect["train_batch_acc"].append(train_acc)
        collect["val_acc"].append(val_acc)
        collect["cost"].append(cost)
        model = model.train()
        #############################################
        # update learning rate
        cur_lr = cyclical_learning_rate(
            batch_step=batch_step,
            step_size=num_epochs * iter_per_ep,
            base_lr=base_lr,
            max_lr=max_lr,
        )
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        ############################################

    print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))

print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))
# torch.stack(collect['train_batch_acc']).cpu().numpy()

plt.plot(
    collect["lr"],
    torch.stack(collect["train_batch_acc"]).cpu().numpy(),
    label="train_batch_acc",
)
plt.plot(collect["lr"], torch.stack(collect["val_acc"]).cpu().numpy(), label="val_acc")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("learning_rate_test.png")
plt.clf()
plt.plot(collect["lr"], torch.stack(collect["cost"]).detach().cpu().numpy())
plt.xlabel("Learning Rate")
plt.ylabel("Current Batch Cost")
plt.savefig("learning_rate_test_cost.png")

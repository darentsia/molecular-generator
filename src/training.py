import torch
from transformers import get_linear_schedule_with_warmup


def configure_optimizer(params, lr):

    optimizer_grouped_parameters = [
        {"params": [p for n, p in params], "lr": lr},
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)

    return optimizer


def configure_scheduler(optimizer, training_steps, warmup):

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=training_steps
    )

    return scheduler

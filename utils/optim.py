from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

def get_optimizer_scheduler(args, model, training_steps):
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.0},
    # ]

    param_optimizer = list(model.named_parameters())
    other_parameters = [(n, p) for n, p in param_optimizer if 'crf' not in n]
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0},
        {'params':[p for n, p in param_optimizer if 'crf.transitions' == n], 'lr':3e-2}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.avg_steps:
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=training_steps,
            num_cycles=int(args.max_epoches/args.avg_steps)
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=training_steps
        )
    
    return optimizer, scheduler
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
import transformers
import timm
import wandb
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from args import get_args
from prior_utils import seed_prior, validate_prior_args, make_prior_model, check_prior_run, check_prior_checkpoint
from layer_replace import layer_replace
from finetune import imagenet_finetune, cifar_finetune, wikitext_finetune, text_embedding_finetune
from plotting import plot_ft
from models import TinyCNN
from modules import ImageNetRNN
from vits import make_vit_tiny, make_vit_small, make_vit_base, make_vit_large
from gpt2_models import gpt2_tiny, gpt2_small

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def theseus():
    args = get_args()
    validate_prior_args(args)
    check_prior_run(args)
    os.makedirs(f'logs/{args.exp_name}', exist_ok = True)
    os.makedirs('saved_models', exist_ok = True)
    args_path = f'logs/{args.exp_name}/training_args.json' if args.optimization_prior and args.reload_ft else f'logs/{args.exp_name}/args.json'
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent = 2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb.init(
        project = 'optimization_prior',
        name = args.exp_name,
        config = args.__dict__
    )
    print(f'RUNNING NETWORK OF THESEUS with {args.model}, {args.pretrained}')
    if args.optimization_prior:
        orig = make_prior_model(args.model, args.teacher_seed, args.teacher_depth)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    elif args.model == 'rn18':
        orig = torchvision.models.resnet18(pretrained = args.pretrained)
    elif args.model == 'vit_tiny':
        orig = make_vit_tiny()
    elif args.model == 'vit_small':
        orig = make_vit_small()
    elif args.model == 'vit_base':
        orig = make_vit_base()
    elif args.model == 'vit_large':
        orig = make_vit_large()
    elif args.model == 'mlp':
        orig = torchvision.models.resnet18(pretrained = False)
    elif args.model == 'vgg':
        orig = torchvision.models.vgg16(pretrained = args.pretrained)
    elif args.model == 'regvgg':
        orig = timm.create_model('timm/repvgg_a2.rvgg_in1k', pretrained = args.pretrained)
    elif args.model == 'dinov2':
        orig = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        if not args.pretrained:
            for m in orig.modules():
                if m.__class__.__name__ in ('MemEffAttention', 'Attention'):
                    for p in m.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)
                        else:
                            nn.init.zeros_(p)
    elif args.model == 'convnext':
        orig = torchvision.models.convnext_tiny(pretrained = args.pretrained)
    elif args.model == 'cnn':
        orig = TinyCNN().to(device)
        orig.load_state_dict(torch.load('saved_models/tiny_cnn_cifar.pt'))
    elif args.model == 'rn50':
        orig = torchvision.models.resnet50(pretrained = args.pretrained)
    elif args.model == 'rn34':
        orig = torchvision.models.resnet34(pretrained = args.pretrained)
    elif args.model == 'rn101':
        orig = torchvision.models.resnet101(pretrained = args.pretrained)
    elif args.model == 'rn152':
        orig = torchvision.models.resnet152(pretrained = args.pretrained)
    elif args.model == 'cnx2_atto':
        orig = timm.create_model('convnextv2_atto.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'cnx2_femto':
        orig = timm.create_model('convnextv2_femto.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'cnx2_pico':
        orig = timm.create_model('convnextv2_pico.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'cnx2_nano':
        orig = timm.create_model('convnextv2_nano.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'cnx2_tiny':
        orig = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'cnx2_base':
        orig = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'cnx2_large':
        orig = timm.create_model('convnextv2_large.fcmae_ft_in1k', pretrained = False)
    elif args.model == 'qwen2.5':
        if args.pretrained:
            orig = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype = torch.bfloat16)
        else:
            config = transformers.AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
            orig = transformers.AutoModelForCausalLM.from_config(config)
        args.model = 'Qwen/Qwen2.5-0.5B'
    elif args.model == 'gpt2':
        if args.pretrained:
            orig = transformers.AutoModelForCausalLM.from_pretrained('gpt2', attn_implementation='eager')
        else:
            config = transformers.AutoConfig.from_pretrained('gpt2')
            orig = transformers.AutoModelForCausalLM.from_config(config, attn_implementation='eager')
    elif args.model == 'gpt2-large':
        if args.pretrained:
            orig = transformers.AutoModelForCausalLM.from_pretrained('gpt2-large')
        else:
            config = transformers.AutoConfig.from_pretrained('gpt2-large')
            orig = transformers.AutoModelForCausalLM.from_config(config)
    elif args.model == 'gpt2-tiny':
        orig = gpt2_tiny()
    elif args.model == 'gpt2-small':
        orig = gpt2_small()
    elif args.model == 'imgrnn':
        orig = ImageNetRNN().to(device)
    elif args.model == 'bert-base':
        if args.pretrained:
            orig = transformers.AutoModel.from_pretrained('bert-base-uncased')
        else:
            config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
            orig = transformers.AutoModel.from_config(config)
    elif args.model == 'bert-large':
        if args.pretrained:
            orig = transformers.AutoModel.from_pretrained('bert-large-uncased')
        else:
            config = transformers.AutoConfig.from_pretrained('bert-large-uncased')
            orig = transformers.AutoModel.from_config(config)
    else:
        raise NotImplementedError
    orig = orig.to(device)
    try:
        orig = orig.to(memory_format = torch.channels_last)
    except Exception:
        pass
    
    if not args.reload_ft and not args.baseline and not args.continue_ft:
        replaced_model_final = layer_replace(args, orig, args.indices, args.replacement_type, low_rank = args.low_rank, rank = args.rank, reload_cka = args.reload_cka, reload_cka_model = args.reload_cka_model)
        torch.save(replaced_model_final.state_dict(), f'saved_models/orig_{args.exp_name}.pt')
    elif args.baseline and not (args.optimization_prior and args.reload_ft):
        if not os.path.exists(f'logs/{args.exp_name}'):
            os.makedirs(f'logs/{args.exp_name}')
        replaced_model_final = layer_replace(args, orig, args.indices, args.replacement_type, low_rank = args.low_rank, rank = args.rank, reload_cka = args.reload_cka, reload_cka_model = args.reload_cka_model)
        replaced_model_final = replaced_model_final.to(device)
    else:
        replaced_model_final = layer_replace(args, orig, args.indices, args.replacement_type, low_rank = args.low_rank, rank = args.rank, reload_ft = args.reload_ft, reload_cka = args.reload_cka, reload_cka_model = args.reload_cka_model)
        print('I am reloading the state dictionary for the model...')
        if args.continue_ft:
            print('I am continuing to finetuning from previous checkpoint')
            replaced_model_final.load_state_dict(torch.load(f'saved_models/ft_{args.exp_name}.pt'))
        else:
            replaced_model_final.load_state_dict(torch.load(f'saved_models/orig_{args.exp_name}.pt', map_location = device, weights_only = True))
            if args.optimization_prior:
                check_prior_checkpoint(args, replaced_model_final)
        replaced_model_final = replaced_model_final.to(device)
    if args.optimization_prior:
        if args.baseline and not args.reload_ft:
            torch.save(replaced_model_final.state_dict(), f'saved_models/orig_{args.exp_name}.pt')
        if args.align_only:
            wandb.finish()
            return
        orig = orig.cpu()
        torch.cuda.empty_cache()
        seed_prior(args.seed + 20000)
        replaced_model_final.requires_grad_(True)
        replaced_model_final.zero_grad(set_to_none = True)
    if args.setting == 'imagenet':
        if args.model == 'dinov2':
            dinov2_im = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
            dinov2_im.backbone = replaced_model_final
            replaced_model_final = dinov2_im
        replaced_model_final, acc, log_json, orig_acc = imagenet_finetune(args, replaced_model_final, orig)
    elif args.setting == 'cifar':
        acc, orig_acc = cifar_finetune(args.exp_name, replaced_model_final, orig)
    elif args.setting == 'embedding':
        replaced_model_final, acc, log_json, orig_acc = text_embedding_finetune(args, replaced_model_final, orig)
    else:
        replaced_model_final, acc, log_json, orig_acc = wikitext_finetune(args, replaced_model_final, orig)
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent = 2)

    with open(f'logs/{args.exp_name}/ft_results_seed{args.seed}.json', 'w') as f:
        json.dump(log_json, f)

    print('Modifed Acc:', acc)
    print('Original Acc:', orig_acc)
    wandb.finish()

if __name__ == '__main__':
    theseus()

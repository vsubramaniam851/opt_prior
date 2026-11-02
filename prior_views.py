import contextlib
import math
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms

def view_spec(args):
    return {'version': 1, 'mode': getattr(args, 'prior_view_mode', 'single'), 'views': 2,
            'crop_scale': [getattr(args, 'prior_view_crop_min', 0.5), 1.0], 'crop_ratio': [0.75, 4 / 3],
            'size': 224, 'flip_probability': 0.5, 'interpolation': 'bilinear', 'antialias': True,
            'validation_seed': getattr(args, 'prior_view_seed', 41000), 'view_batching': 'separate',
            'normalization_mean': [0.485, 0.456, 0.406], 'normalization_std': [0.229, 0.224, 0.225]}

def view_transform(args):
    spec = view_spec(args)
    return transforms.Compose([transforms.RandomResizedCrop(spec['size'], scale = tuple(spec['crop_scale']),
                               ratio = tuple(spec['crop_ratio']), interpolation = transforms.InterpolationMode.BILINEAR,
                               antialias = True), transforms.RandomHorizontalFlip(spec['flip_probability']),
                               transforms.ToTensor(), transforms.Normalize(spec['normalization_mean'], spec['normalization_std'])])

class PriorViewDataset(Dataset):
    def __init__(self, dataset, transform, seed = None):
        self.dataset = dataset
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        views = []
        for view in range(2):
            context = torch.random.fork_rng(devices = []) if self.seed is not None else contextlib.nullcontext()
            with context:
                if self.seed is not None:
                    torch.random.default_generator.manual_seed((self.seed + 2 * index + view) % (2 ** 63 - 1))
                views.append(self.transform(image))
        return tuple(views), label

def view_images(images, device):
    if isinstance(images, torch.Tensor):
        return images.to(device, non_blocking = True)
    if not isinstance(images, (tuple, list)) or len(images) != 2 or not all(isinstance(x, torch.Tensor) for x in images):
        raise ValueError('Expected an image tensor or two matched view tensors')
    if images[0].shape != images[1].shape:
        raise ValueError('Both view tensors must contain the same ordered image batch')
    return tuple(x.to(device, non_blocking = True) for x in images)

def view_batch_size(images):
    return images.size(0) if isinstance(images, torch.Tensor) else images[0].size(0)

def view_kernel(x, unbiased = True):
    if x.dim() != 2 or x.size(0) < (4 if unbiased else 2):
        raise ValueError('Invalid feature matrix for view CKA')
    with torch.autocast(device_type = x.device.type, enabled = False):
        x = x.double() if x.dtype == torch.float64 else x.float()
        x = x - x.mean(dim = 0, keepdim = True)
        x = x / x.square().mean().sqrt().detach().clamp_min(1e-12)
        gram = x @ x.T / x.size(1)
        if unbiased:
            n = x.size(0)
            mask = ~torch.eye(n, device = x.device, dtype = torch.bool)
            gram = gram * mask
            gram = (gram - gram.sum(dim = 0, keepdim = True) / (n - 2)
                    - gram.sum(dim = 1, keepdim = True) / (n - 2) + gram.sum() / ((n - 1) * (n - 2))) * mask
        return gram / torch.linalg.vector_norm(gram).clamp_min(1e-12)

def view_log_neighbors(x, k, temperature):
    if x.dim() != 2 or not 1 <= k < x.size(0) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid soft-neighborhood matrix, k or temperature')
    with torch.autocast(device_type = x.device.type, enabled = False):
        x = x.double() if x.dtype == torch.float64 else x.float()
        x = F.normalize(x - x.mean(dim = 0, keepdim = True), dim = 1)
        n = x.size(0)
        mask = ~torch.eye(n, device = x.device, dtype = torch.bool)
        similarity = (x @ x.T)[mask].reshape(n, n - 1)
        similarity = similarity - similarity.mean(dim = 1, keepdim = True)
        similarity = similarity / similarity.square().mean(dim = 1, keepdim = True).clamp_min(1e-8).sqrt()
        threshold = similarity.topk(k, dim = 1).values[:, -1:]
        weights = F.logsigmoid((similarity - threshold) / temperature)
        return weights - torch.logsumexp(weights, dim = 1, keepdim = True)

def view_similarity(a, b):
    a = a / torch.linalg.vector_norm(a).clamp_min(1e-12)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-12)
    return (a * b).sum().clamp(-1.0, 1.0)

def view_target(targets, args):
    with torch.no_grad():
        if args.repdist == 'KNN':
            values = [view_log_neighbors(y.detach(), args.knn_k, args.knn_temp) for y in targets]
            shared = torch.logaddexp(values[0], values[1]) - math.log(2)
        else:
            unbiased = args.repdist == 'UCKA' or args.cka_loss == 'debiased'
            values = [view_kernel(y.detach(), unbiased) for y in targets]
            shared = (values[0] + values[1]) / 2
            if torch.linalg.vector_norm(shared).item() <= 1e-12:
                raise FloatingPointError('The shared guide CKA target has zero norm')
        return values, shared

def view_loss(features, targets, args, mode = None):
    mode = getattr(args, 'prior_view_mode', 'separate') if mode is None else mode
    if mode not in ['separate', 'shared'] or len(features) != 2 or len(targets) != 2:
        raise ValueError('Two-view loss requires separate or shared mode and two feature/target matrices')
    if any(x.dim() != 2 or x.size(0) != features[0].size(0) for x in list(features) + list(targets)):
        raise ValueError('Two-view features must have matching image counts')
    with torch.autocast(device_type = features[0].device.type, enabled = False):
        individual, shared = view_target(targets, args)
        losses = []
        for index, x in enumerate(features):
            target = shared if mode == 'shared' else individual[index]
            if args.repdist == 'KNN':
                log_p = view_log_neighbors(x, args.knn_k, args.knn_temp)
                losses.append((target.exp() * (target - log_p)).sum(dim = 1).mean())
            else:
                kernel = view_kernel(x, args.repdist == 'UCKA' or args.cka_loss == 'debiased')
                losses.append(1.0 - view_similarity(kernel, target))
        return torch.stack(losses).mean()

def view_overlap(a, b, k):
    def neighbors(x):
        x = F.normalize(x.float() - x.float().mean(dim = 0, keepdim = True), dim = 1)
        mask = torch.eye(x.size(0), device = x.device, dtype = torch.bool)
        return (x @ x.T).masked_fill(mask, -torch.inf).topk(k, dim = 1).indices
    a, b = neighbors(a), neighbors(b)
    return (a.unsqueeze(2) == b.unsqueeze(1)).any(dim = 2).float().mean().item()

def view_diagnostics(features, targets, args):
    with torch.no_grad(), torch.autocast(device_type = features[0].device.type, enabled = False):
        student = [view_kernel(x) for x in features]
        guide = [view_kernel(y) for y in targets]
        shared_kernel = (guide[0] + guide[1]) / 2
        log_q = [view_log_neighbors(y, args.knn_k, args.knn_temp) for y in targets]
        shared_log_q = torch.logaddexp(*log_q) - math.log(2)
        entropy = -(shared_log_q.exp() * shared_log_q).sum(dim = 1)
        js = sum((q.exp() * (q - shared_log_q)).sum(dim = 1).mean() for q in log_q) / 2
        n = features[0].size(0)
        return {'student_cross_view_ucka': view_similarity(*student).item(),
                'guide_cross_view_ucka': view_similarity(*guide).item(),
                'student_cross_view_knn_overlap': view_overlap(*features, args.knn_k),
                'guide_cross_view_knn_overlap': view_overlap(*targets, args.knn_k),
                'guide_permuted_view_ucka': view_similarity(guide[0], guide[1].roll((1, 1), dims = (0, 1))).item(),
                'guide_kernel_coherence': torch.linalg.vector_norm(shared_kernel).item(),
                'student_shared_target_ucka': sum(view_similarity(x, shared_kernel).item() for x in student) / 2,
                'target_neighbor_entropy': entropy.mean().item(),
                'target_neighbor_entropy_normalized': (entropy / math.log(n - 1)).mean().item(),
                'target_effective_neighbors': entropy.exp().mean().item(),
                'guide_neighbor_js': js.item(),
                'separate_view_loss': view_loss(features, targets, args, 'separate').item(),
                'shared_view_loss': view_loss(features, targets, args, 'shared').item()}

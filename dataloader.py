import os
import numpy as np
import glob
import random
import torch
import torch.utils.data as data
import transformers
import datasets
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torchvision
import datasets

from collections import deque
from itertools import islice
from pytorch_pretrained_bert import BertTokenizer
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from tqdm import tqdm

IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

def ascii_digits_to_int64(label_bytes):
    s = bytes(label_bytes.tolist()).decode('ascii')
    return np.array(int(s), dtype=np.int64)

def create_imagenet_dataloader(dataset_name, split, batch_size, data_dir, seed = 12):
    if split == 'train':
        is_train = True
        split_dir = 'train'
    else:
        is_train = False
        split_dir = 'val'
    num_threads = 16
    device_id = 0
    seed = seed + device_id

    if dataset_name == 'imagenet-1k':
        data_dir = os.path.join(data_dir, split_dir)
        if is_train:
            pipe = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id, seed = seed)
            with pipe:
                jpegs, labels = fn.readers.file(name = 'Reader', file_root = data_dir, random_shuffle = True, pad_last_batch = True)
                images = fn.decoders.image_random_crop(jpegs, device = 'mixed', output_type = types.RGB,
                                                    random_area = [0.08, 1.0], random_aspect_ratio = [0.75, 1.333])
                images = fn.resize(images, resize_x = 224, resize_y = 224)
                mirror = fn.random.coin_flip(probability = 0.5)
                images = fn.crop_mirror_normalize(images, device = 'gpu', dtype = types.FLOAT, output_layout = types.NHWC, crop = (224, 224),
                                                  mean = IMAGENET_MEAN, std = IMAGENET_STD, mirror = mirror)
                labels = labels.gpu()
                pipe.set_outputs(images, labels)
        else:
            pipe = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id, seed = seed)
            with pipe:
                jpegs, labels = fn.readers.file(name = 'Reader', file_root = data_dir, random_shuffle = False, pad_last_batch = True)
                images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
                images = fn.resize(images, resize_shorter=256)
                images = fn.crop_mirror_normalize(images, device = 'gpu', dtype = types.FLOAT, output_layout = types.NHWC, crop = (224, 224),
                                                  mean = IMAGENET_MEAN, std = IMAGENET_STD)
    elif dataset_name == 'imagenet-21k':
        shard_id = 0
        num_shards = 1
        tar_paths = sorted(glob.glob(os.path.join(data_dir, '*.tar')))
        if not tar_paths:
            raise FileNotFoundError(f'No .tar shards found under: {wds_root}')
        idx_paths = [p[:-4] + '.idx' for p in tar_paths]
        missing = [p for p in idx_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f'Missing .idx files. Example: {missing[0]}')
        pipe = Pipeline(batch_size = batch_size, num_threads = num_threads, device_id = device_id, seed = seed + device_id, prefetch_queue_depth = 2)
        with pipe:
            img_raw, cls_bytes = fn.readers.webdataset(paths = tar_paths, index_paths=idx_paths, ext=['jpg', 'cls'], 
                                                       missing_component_behavior = 'error', random_shuffle = is_train, 
                                                       initial_fill=4096 if is_train else 0, shard_id = shard_id, num_shards = num_shards, 
                                                       stick_to_shard = True, dont_use_mmap = True, name = 'Reader')

            images = fn.decoders.image(img_raw, device = 'mixed', output_type = types.RGB)
            if is_train:
                images = fn.random_resized_crop(images, size=(224, 224))
                mirror = fn.random.coin_flip(probability=0.5)
                images = fn.crop_mirror_normalize(images, device = 'gpu', dtype = types.FLOAT, output_layout = types.NHWC, mean = IMAGENET_MEAN,
                                                  std = IMAGENET_STD, mirror = mirror)
            else:
                images = fn.resize(images, resize_shorter=256)
                images = fn.crop_mirror_normalize(images, device = 'gpu', dtype = types.FLOAT, output_layout = types.NHWC, crop = (224, 224),
                                                  mean = IMAGENET_MEAN, std = IMAGENET_STD)
            labels = fn.python_function(cls_bytes, function = ascii_digits_to_int64, num_outputs = 1)
            labels = fn.cast(labels, dtype = types.INT64)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    labels = labels.gpu()
    pipe.set_outputs(images, labels)
    pipe.build()

    output_map = ['images', 'labels']
    dali_iter = DALIGenericIterator([pipe], output_map = output_map, reader_name = 'Reader', last_batch_policy = LastBatchPolicy.DROP if is_train else LastBatchPolicy.PARTIAL, 
                                    auto_reset = True)
    return dali_iter

def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y, None, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample(()).to(device=x.device)
    perm = torch.randperm(x.size(0), device=x.device)

    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return x_mix, y_a, y_b, lam

def get_imagenet_dataloaders(dataset_name = 'imagenet-1k', batch_size = 256, data_dir = '../../imagenet_pytorch', seed = 12):
    train_loader = create_imagenet_dataloader(dataset_name, 'train', batch_size, data_dir, seed = seed)
    val_loader = create_imagenet_dataloader(dataset_name, 'val', batch_size, data_dir, seed = seed)
    return train_loader, val_loader

def cifar_dataset(batch_size = 256):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tfms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean, std)])
    test_tfms  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR10(train=True,  download=True, transform=train_tfms)
    test_ds  = torchvision.datasets.CIFAR10(train=False, download=True, transform=test_tfms)
    train_loader = data.DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = data.DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

class CocoSegmentationDataset(data.Dataset):
    def __init__(self, root, split = 'train2017', size = 224, image_mean = (0.485, 0.456, 0.406),
                 image_std  = (0.229, 0.224, 0.225)):
        self.root = root
        self.images_dir = os.path.join(root, 'images', split)
        self.masks_dir  = os.path.join(root, 'masks', split)

        self.files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(image_mean, image_std),
        ])
        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size, interpolation=Image.NEAREST),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.PILToTensor(),  # [1, H, W]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.images_dir, fname)
        stem, _ = os.path.splitext(fname)

        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(self.masks_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(f'No mask found for {img_path}')

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        img  = self.image_transform(img)
        mask = self.mask_transform(mask).squeeze(0).long()  # [H, W]

        return img, mask

def get_coco_segmentation_dataloaders(root, batch_size = 8, size = 224, num_workers = 4,
                                     train_split = 'train2017', val_split = 'val2017'):
    train_ds = CocoSegmentationDataset(
        root=root, split=train_split, size=size,
    )
    val_ds   = CocoSegmentationDataset(
        root=root, split=val_split, size=size,
    )

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader   = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader

def generalization_cifar_test(batch_size = 256, imagenet_style = True, size = 224):
    if imagenet_style:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        train_tfms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.RandomCrop(size, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        test_tfms  = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        train_tfms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
        test_tfms  = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])

    train_ds = torchvision.datasets.CIFAR100(root = '../.cache', train=True,  download=True, transform=train_tfms)
    test_ds  = torchvision.datasets.CIFAR100(root = '../.cache', train=False, download=True, transform=test_tfms)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader = data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

class StreamingTextDataset(data.IterableDataset):
    def __init__(self, hf_streaming_ds, tokenizer, *, text_key = 'text', seq_len = 1024, tokenize_batch_size = 2048, add_eos_between_docs = True, eos_token_id = None):
        super().__init__()
        self.hf_ds = hf_streaming_ds
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.seq_len = seq_len
        self.tokenize_batch_size = tokenize_batch_size
        self.add_eos_between_docs = add_eos_between_docs

        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_token_id is None:
            eos_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]

        self.eos_token_id = int(eos_token_id)

    def _iter_shard(self):
        info = data.get_worker_info()
        if info is None:
            return iter(self.hf_ds)
        return iter(self.hf_ds.shard(num_shards = info.num_workers, index = info.id))

    def _yield_chunks_from_buffer(self, buf):
        while len(buf) >= self.seq_len:
            chunk = [buf.popleft() for _ in range(self.seq_len)]
            yield chunk

    def _consume_and_chunk(self, texts, token_buffer):
        enc = self.tokenizer(texts, add_special_tokens = False, return_attention_mask = False, return_token_type_ids = False, truncation = False)
        input_ids_batch = enc['input_ids']
        if self.add_eos_between_docs:
            for ids in input_ids_batch:
                token_buffer.extend(ids)
                token_buffer.append(self.eos_token_id)
        else:
            for ids in input_ids_batch:
                token_buffer.extend(ids)

    def _make_example(self, chunk_ids):
        input_ids = torch.tensor(chunk_ids, dtype = torch.long)
        attn = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attn,
            'labels': input_ids.clone()
        }

    def __iter__(self):
        token_buffer = deque()
        pending_texts = []
        
        for ex in self._iter_shard():
            txt = ex[self.text_key]
            if not isinstance(txt, str):
                txt = str(txt)
            pending_texts.append(txt)

        if len(pending_texts) >= self.tokenize_batch_size:
            self._consume_and_chunk(pending_texts, token_buffer)
            pending_texts.clear()

            for chunk in self._yield_chunks_from_buffer(token_buffer):
                yield self._make_example(chunk)

        if pending_texts:
            self._consume_and_chunk(pending_texts, token_buffer)
            pending_texts.clear()
        
        for chunk in self._yield_chunks_from_buffer(token_buffer):
            yield self.make_example(chunk)

class SizedLoader:
    def __init__(self, loader, steps_per_epoch):
        self.loader = loader
        self.steps = int(steps_per_epoch)

    def __iter__(self):
        return iter(islice(self.loader, self.steps))

    def __len__(self):
        return self.steps

def build_text_dataloader(split, tokenizer, *, seq_len = 1024, batch_size = 64, num_workers = 16, shuffle = True, shuffle_buffer = 10_000, text_key = 'text', 
                            tokenize_batch_size = 2048, add_eos_between_docs = True, drop_last = True):
    ds = datasets.load_dataset('DKYoon/SlimPajama-6B', split = split, streaming = True)

    if shuffle:
        ds = ds.shuffle(buffer_size = shuffle_buffer)
    iter_ds = StreamingTextDataset(ds, tokenizer, text_key = text_key, seq_len = seq_len, tokenize_batch_size = tokenize_batch_size, add_eos_between_docs = add_eos_between_docs,
                                    eos_token_id = getattr(tokenizer, 'eos_token_ids', None))

    def collate(features):
        return {
            'input_ids': torch.stack([f['input_ids'] for f in features], dim=0),
            'attention_mask': torch.stack([f['attention_mask'] for f in features], dim=0),
            'labels': torch.stack([f['labels'] for f in features], dim=0),
        }
    
    if num_workers > 0:
        persistent_workers = True
        prefetch_factor = 2

    else:
        persistent_workers = False
        prefetch_factor = 1

    return data.DataLoader(iter_ds, batch_size = batch_size, collate_fn = collate, num_workers = num_workers, pin_memory = True, drop_last = drop_last,
                            persistent_workers = persistent_workers, prefetch_factor = prefetch_factor)

def sj_get_text_dataloaders(model_name, seq_len = 1024, batch_size = 64, num_workers = 16, tokenize_batch_size = 2048, shuffle_buffer = 10_000):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloaders = []
    for split in ['train', 'validation']:
        dl = build_text_dataloader(split, tokenizer, seq_len = seq_len, batch_size = batch_size, num_workers = num_workers, shuffle = (split == 'train'), 
                                    tokenize_batch_size = tokenize_batch_size, shuffle_buffer = shuffle_buffer)
        dataloaders.append(dl)

    return dataloaders

def _format_alpaca(batch):
    texts = []
    for instr, inp, output in zip(batch.get('instruction', []), batch.get('input', []), batch.get('output', [])):
        if inp and len(inp.strip()) > 0:
            t = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{output}'
        else:
            t = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n ### Instruction:\n{instr}\n\n### Response:\n{output}'
        texts.append(t)
    return {'text': texts}

def _tokenize_then_group(ds, tokenizer, seq_len, num_proc = 8):
    eos = tokenizer.eos_token or ''

    def tok(batch):
        texts = [x + eos for x in batch['text']]
        out = tokenizer(texts, add_special_tokens = False, return_attention_mask = True)
        return out

    tokenized = ds.map(tok, batched = True, remove_columns = ds.column_names, num_proc = num_proc)

    def group(batch):
        input_ids = sum(batch['input_ids'], [])
        attn = sum(batch['attention_mask'], [])
        total = (len(input_ids) // seq_len) * seq_len
        input_ids = input_ids[:total]
        attn = attn[:total]

        out = {'input_ids': [input_ids[i:i+seq_len] for i in range(0, total, seq_len)],
               'attention_mask': [attn[i:i+seq_len] for i in range(0, total, seq_len)]}
        return out

    chunked = tokenized.map(group, batched = True, remove_columns = tokenized.column_names, num_proc = num_proc)
    return chunked.with_format(type = 'torch', columns = ['input_ids', 'attention_mask'])

def _prepare_hf_dataset(model_name, seq_len = 1024, val_fraction = 0.02, num_proc = 8):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast = True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_size = 'left'

    raw = datasets.load_dataset('yahma/alpaca-cleaned')['train']
    formatted = raw.map(_format_alpaca, batched = True, remove_columns = raw.column_names, num_proc = num_proc, load_from_cache_file=False)

    formatted = formatted.shuffle()
    ds_dict = formatted.train_test_split(test_size = val_fraction)
    train = _tokenize_then_group(ds_dict['train'], tokenizer, seq_len, num_proc)
    val = _tokenize_then_group(ds_dict['test'], tokenizer, seq_len, num_proc)
    return tokenizer, datasets.DatasetDict(train = train, test = val)

def get_alpaca_text_dataloaders(model, batch_size = 256, seq_len = 1024, num_workers = 16, *, val_fraction = 0.02, num_proc = 8):
    _, ds = _prepare_hf_dataset(model_name = model, seq_len = seq_len, val_fraction = val_fraction, num_proc = num_proc)

    train_loader = data.DataLoader(ds['train'], batch_size = batch_size, shuffle = True, drop_last = True, num_workers = num_workers, pin_memory = True,
                                   persistent_workers = (num_workers > 0), prefetch_factor = 2 if num_workers > 0 else None)

    val_loader = data.DataLoader(ds['test'], batch_size = batch_size, shuffle = False, drop_last = False, num_workers = num_workers, pin_memory = True,
                                 persistent_workers = (num_workers > 0), prefetch_factor = 2 if num_workers > 0 else None)

    return train_loader, val_loader

def _dict_has_sequence(value):
    return any(isinstance(v, (list, tuple)) for v in value.values())

def _dict_index(value, idx):
    out = {}
    for k, v in value.items():
        if isinstance(v, (list, tuple)):
            out[k] = v[idx] if len(v) > idx else ''
        else:
            out[k] = v
    return out

def _extract_text_field(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if _dict_has_sequence(value):
            lens = [len(v) for v in value.values() if isinstance(v, (list, tuple))]
            if len(lens) == 0 or max(lens) == 0:
                return ''
            return _extract_text_field(_dict_index(value, 0))
        title = value.get('title', '') or ''
        text = value.get('text', '') or value.get('contents', '') or value.get('passage', '') or ''
        title = str(title)
        text = str(text)
        if len(title) > 0 and len(text) > 0:
            return title + ' ' + text
        return title or text
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ''
        return _extract_text_field(value[0])
    return str(value)

def _sample_text_field(value):
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return _extract_text_field(random.choice(value))
    if isinstance(value, dict) and _dict_has_sequence(value):
        lens = [len(v) for v in value.values() if isinstance(v, (list, tuple))]
        if len(lens) > 0 and max(lens) > 0:
            return _extract_text_field(_dict_index(value, random.randrange(max(lens))))
    return _extract_text_field(value)

def _normalize_dataset_config(dataset_name, dataset_config):
    if dataset_config is None:
        return None
    dataset_config = str(dataset_config)
    if dataset_config.lower() in ['', 'none', 'null']:
        return None
    if dataset_config.lower() == 'auto':
        if dataset_name == 'sentence-transformers/msmarco-msmarco-MiniLM-L6-v3':
            return 'triplet-hard'
        return None
    return dataset_config

def _load_dataset_with_config(dataset_name, dataset_config, split, streaming = False):
    dataset_config = _normalize_dataset_config(dataset_name, dataset_config)
    if dataset_config is None:
        return datasets.load_dataset(dataset_name, split = split, streaming = streaming)
    return datasets.load_dataset(dataset_name, dataset_config, split = split, streaming = streaming)

def _slice_split(split, max_samples):
    if max_samples is None:
        return split
    if '[' in split:
        return split
    return f'{split}[:{int(max_samples)}]'

def _materialize_streaming_dataset(ds, max_samples):
    if max_samples is None:
        max_samples = 100000
    max_samples = int(max_samples)
    rows = list(islice(ds, max_samples))
    if len(rows) == 0:
        raise ValueError('Streaming dataset produced no rows')
    return datasets.Dataset.from_list(rows)

def _stream_embedding_dataset(dataset_name, dataset_config, split, max_samples = None, seed = 0):
    stream = _load_dataset_with_config(dataset_name, dataset_config, split, streaming = True)
    try:
        stream = stream.shuffle(buffer_size = 10000, seed = seed)
    except Exception:
        pass
    return _materialize_streaming_dataset(stream, max_samples)

def _load_embedding_dataset(dataset_name, dataset_config, split, max_samples = None, seed = 0, streaming_fallback = True):
    try:
        return _load_dataset_with_config(dataset_name, dataset_config, split)
    except Exception as e:
        if not streaming_fallback:
            raise
        print(f'Falling back to streaming load for {dataset_name} split={split} after dataset load error: {e}')
        return _stream_embedding_dataset(dataset_name, dataset_config, split, max_samples = max_samples, seed = seed)

class TextEmbeddingDataset(data.Dataset):
    def __init__(self, hf_dataset, negatives_per_query = 1):
        self.dataset = hf_dataset
        self.negatives_per_query = negatives_per_query

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[int(idx)]
        query = _extract_text_field(ex.get('query', ex.get('question', ex.get('queries', ex.get('text', '')))))
        positive = _sample_text_field(ex.get('positive_passages', ex.get('positive', ex.get('pos', ex.get('passages', ex.get('passage', ''))))))
        negatives = ex.get('negative_passages', ex.get('negative', ex.get('neg', [])))
        if negatives == []:
            negative_keys = sorted([k for k in ex.keys() if k.startswith('negative_')])
            negatives = [ex[k] for k in negative_keys]

        neg_texts = []
        if isinstance(negatives, (list, tuple)):
            if len(negatives) > 0:
                if self.negatives_per_query <= 0:
                    neg_texts = []
                else:
                    k = min(self.negatives_per_query, len(negatives))
                    neg_texts = [_extract_text_field(x) for x in random.sample(list(negatives), k)]
        elif isinstance(negatives, dict) and _dict_has_sequence(negatives):
            lens = [len(v) for v in negatives.values() if isinstance(v, (list, tuple))]
            if len(lens) > 0 and max(lens) > 0 and self.negatives_per_query > 0:
                k = min(self.negatives_per_query, max(lens))
                neg_texts = [_extract_text_field(_dict_index(negatives, j)) for j in random.sample(range(max(lens)), k)]
        else:
            neg = _extract_text_field(negatives)
            if len(neg) > 0 and self.negatives_per_query > 0:
                neg_texts = [neg]

        if len(query) == 0:
            query = positive
        if len(positive) == 0:
            positive = query

        return {'query': query, 'positive': positive, 'negatives': neg_texts}

class TextEmbeddingCollator:
    def __init__(self, tokenizer, query_max_len = 32, passage_max_len = 192):
        self.tokenizer = tokenizer
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def _tok(self, texts, max_len):
        return self.tokenizer(texts, padding = True, truncation = True, max_length = max_len, return_tensors = 'pt')

    def __call__(self, features):
        queries = [f['query'] for f in features]
        positives = [f['positive'] for f in features]
        negatives = []
        negative_counts = []
        for f in features:
            negative_counts.append(len(f['negatives']))
            negatives.extend(f['negatives'])

        query_tok = self._tok(queries, self.query_max_len)
        positive_tok = self._tok(positives, self.passage_max_len)
        align_tok = self._tok(queries + positives, max(self.query_max_len, self.passage_max_len))
        batch = {
            'input_ids': align_tok['input_ids'],
            'attention_mask': align_tok['attention_mask'],
            'query_input_ids': query_tok['input_ids'],
            'query_attention_mask': query_tok['attention_mask'],
            'positive_input_ids': positive_tok['input_ids'],
            'positive_attention_mask': positive_tok['attention_mask'],
            'negative_counts': torch.tensor(negative_counts, dtype = torch.long),
            'query_texts': queries,
            'positive_texts': positives,
        }
        if 'token_type_ids' in align_tok:
            batch['token_type_ids'] = align_tok['token_type_ids']
        if 'token_type_ids' in query_tok:
            batch['query_token_type_ids'] = query_tok['token_type_ids']
        if 'token_type_ids' in positive_tok:
            batch['positive_token_type_ids'] = positive_tok['token_type_ids']
        if len(negatives) > 0:
            negative_tok = self._tok(negatives, self.passage_max_len)
            batch['negative_input_ids'] = negative_tok['input_ids']
            batch['negative_attention_mask'] = negative_tok['attention_mask']
            if 'token_type_ids' in negative_tok:
                batch['negative_token_type_ids'] = negative_tok['token_type_ids']
            batch['negative_texts'] = negatives
        else:
            batch['negative_texts'] = []
        return batch

def _maybe_select(ds, max_samples):
    if max_samples is None:
        return ds
    max_samples = min(int(max_samples), len(ds))
    return ds.select(range(max_samples))

def get_text_embedding_dataloaders(model, batch_size, num_workers = 16, seq_len = 512, dataset_name = 'sentence-transformers/msmarco-msmarco-MiniLM-L6-v3',
                                   dataset_config = 'auto', train_split = 'train', val_split = None, val_fraction = 0.01, max_train_samples = None,
                                   max_val_samples = 4096, query_max_len = 32, passage_max_len = 192, negatives_per_query = 1, seed = 0,
                                   streaming_train_samples = 100000):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if val_split is not None:
        stream_train_cap = max_train_samples if max_train_samples is not None else streaming_train_samples
        train_load_split = _slice_split(train_split, max_train_samples)
        val_load_split = _slice_split(val_split, max_val_samples)
        train_raw = _load_embedding_dataset(dataset_name, dataset_config, train_load_split, max_samples = stream_train_cap, seed = seed)
        val_raw = _load_embedding_dataset(dataset_name, dataset_config, val_load_split, max_samples = max_val_samples, seed = seed + 1)
    else:
        try:
            if max_train_samples is not None:
                val_cap = max_val_samples if max_val_samples is not None else max(1, int(max_train_samples * val_fraction))
                raw_cap = int(max_train_samples) + int(val_cap)
            else:
                raw_cap = None
            raw_load_split = _slice_split(train_split, raw_cap)
            raw = _load_embedding_dataset(dataset_name, dataset_config, raw_load_split, max_samples = None, seed = seed, streaming_fallback = False)
            if max_train_samples is not None and max_val_samples is not None:
                test_size = min(int(max_val_samples), max(1, len(raw) - 1))
            else:
                test_size = val_fraction
            split = raw.train_test_split(test_size = test_size, seed = seed)
            train_raw = split['train']
            val_raw = split['test']
        except Exception as e:
            stream_train_cap = max_train_samples if max_train_samples is not None else streaming_train_samples
            stream_val_cap = max_val_samples if max_val_samples is not None else 4096
            stream_cap = int(stream_train_cap) + int(stream_val_cap)
            print(f'Falling back to streaming train/val materialization for {dataset_name} after dataset load error: {e}')
            raw = _stream_embedding_dataset(dataset_name, dataset_config, train_split, max_samples = stream_cap, seed = seed)
            train_end = min(int(stream_train_cap), len(raw))
            val_end = min(train_end + int(stream_val_cap), len(raw))
            if val_end <= train_end:
                train_end = max(1, int(len(raw) * (1.0 - val_fraction)))
                val_end = len(raw)
            train_raw = raw.select(range(train_end))
            val_raw = raw.select(range(train_end, val_end))

    train_raw = _maybe_select(train_raw, max_train_samples)
    val_raw = _maybe_select(val_raw, max_val_samples)

    train_ds = TextEmbeddingDataset(train_raw, negatives_per_query = negatives_per_query)
    val_ds = TextEmbeddingDataset(val_raw, negatives_per_query = negatives_per_query)
    collator = TextEmbeddingCollator(tokenizer, query_max_len = query_max_len, passage_max_len = passage_max_len)

    loader_kwargs = {'persistent_workers': (num_workers > 0)}
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2

    train_loader = data.DataLoader(train_ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = num_workers, pin_memory = True,
                                   collate_fn = collator, **loader_kwargs)
    val_loader = data.DataLoader(val_ds, batch_size = batch_size, shuffle = False, drop_last = False, num_workers = num_workers, pin_memory = True,
                                 collate_fn = collator, **loader_kwargs)
    return train_loader, val_loader

class TextDataset(data.Dataset):
    def __init__(self, dataset, num_words):
        self.dataset = dataset
        self.num_words = num_words
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.dataset[idx], 'target_ids': self.dataset[idx]}

def get_text_dataloaders(model, batch_size, num_workers = 32, seq_len = 1024):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, do_lower_case = False)
    dataloaders = []
    file_path = 'wikitext-103/wiki.{}.tokens'
    if not os.path.exists(os.path.join('cached_wikitext')):
        os.makedirs(os.path.join('cached_wikitext'))
    cached_file_path = os.path.join('cached_wikitext', 'wikitext_{}.pt')
    for split in ['train', 'valid', 'test']:
        if os.path.exists(cached_file_path.format(split)):
            wikitext_dataset = torch.load(cached_file_path.format(split))
            num_words = wikitext_dataset['num_words']
            dataset = wikitext_dataset['dataset']
        else:
            with open(file_path.format(split), 'r', encoding = 'utf-8') as f:
                dataset = f.readlines()
                num_words = sum([len(line.split()) for line in dataset])
            dataset = list(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]'))) for line in tqdm(dataset))
            dataset = torch.tensor([index for line in dataset for index in line], dtype = torch.long)
            torch.save({'dataset': dataset, 'num_words': num_words}, cached_file_path.format(split))
        num_sequences = (dataset.size(0) // seq_len) * seq_len
        dataset = dataset.narrow(0, 0, num_sequences).view(-1, seq_len)
        dataset = TextDataset(dataset, num_words)
        dataloader = data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle = (split == 'train'))
        dataloaders.append(dataloader)
    return dataloaders[0], dataloaders[1], dataloaders[2], len(tokenizer.vocab)
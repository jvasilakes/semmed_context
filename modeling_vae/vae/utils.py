import os
import pdb
import json
import pickle
import random
import logging
import traceback
from collections import defaultdict

import torch
import numpy as np


from vae.data.datamodules import min_pad_collate


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AutogradDebugger(torch.autograd.detect_anomaly):

    def __init__(self):
        super(AutogradDebugger, self).__init__()

    def __enter__(self):
        super(AutogradDebugger, self).__enter__()
        return self

    def __exit__(self, type, value, trace):
        super(AutogradDebugger, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            self.halt(str(value))

    @staticmethod
    def halt(msg):
        print()
        print("==========================================")
        print("     Failure! Left mouse to continue.")
        print("==========================================")
        print()
        print(msg)
        pdb.set_trace()


def load_glove(path):
    """
    Load the GLoVe embeddings from the provided path.
    Return the embedding matrix and the embedding dimension.
    Pickles the loaded embedding matrix for fast loading
    in the future.

    :param str path: Path to the embeddings. E.g.
                     `glove.6B/glove.6B.100d.txt`
    :returns: embeddings, embedding_dim
    :rtype: Tuple(numpy.ndarray, int)
    """
    bn = os.path.splitext(os.path.basename(path))[0]
    pickle_file = bn + ".pickle"
    if os.path.exists(pickle_file):
        logging.warning(f"Loading embeddings from pickle file {pickle_file} in current directory.")  # noqa
        glove = pickle.load(open(pickle_file, "rb"))
        emb_dim = list(glove.values())[0].shape[0]
        return glove, emb_dim

    vectors = []
    words = []
    idx = 0
    word2idx = {}

    with open(path, "rb") as inF:
        for line in inF:
            line = line.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    emb_dim = vect.shape[0]
    glove = {word: np.array(vectors[word2idx[word]]) for word in words}
    if not os.path.exists(pickle_file):
        pickle.dump(glove, open(pickle_file, "wb"))
    return glove, emb_dim


def get_embedding_matrix(vocab, glove):
    emb_dim = len(list(glove.values())[0])
    matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    found = 0
    for (i, word) in enumerate(vocab):
        try:
            matrix[i] = glove[word]
            found += 1
        except KeyError:
            matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    logging.info(f"Found {found}/{len(vocab)} vocab words in embedding.")
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    return matrix, word2idx


def load_latest_checkpoint(model, optimizer, checkpoint_dir,
                           map_location=None):
    """
    Find the most recent (in epochs) checkpoint in checkpoint dir and load
    it into the model and optimizer. Return the model and optimizer
    along with the epoch the checkpoint was trained to.
    If not checkpoint is found, return the unchanged model and optimizer,
    and 0 for the epoch.
    """
    ls = os.listdir(checkpoint_dir)
    ckpts = [fname for fname in ls if fname.endswith(".pt")]
    if len(ckpts) == 0:
        return model, optimizer, 0, None
    else:
        latest_ckpt_idx = 0
        latest_epoch = 0
        for (i, ckpt) in enumerate(ckpts):
            epoch = ckpt.replace("model_", '').replace(".pt", '')
            epoch = int(epoch)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt_idx = i

    ckpt = torch.load(os.path.join(checkpoint_dir, ckpts[latest_ckpt_idx]),
                      map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    next_epoch = ckpt["epoch"] + 1
    return model, optimizer, next_epoch, ckpts[latest_ckpt_idx]


def pad_sequence(batch):
    """
    Pad the sequence batch with 0 vectors.
    Compute the original lengths of each sequence.
    Collect labels into a dict of tensors.
    Meant to be used as the value of the `collate_fn`
    argument to `torch.utils.data.DataLoader`.
    """
    seqs = [torch.squeeze(x) for (x, _) in batch]
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    lengths = torch.LongTensor([len(s) for s in seqs])
    labels = defaultdict(list)
    for (_, y) in batch:
        for label_name in y.keys():
            labels[label_name].append(y[label_name])
    for label_name in labels.keys():
        labels[label_name] = torch.stack(labels[label_name])
    return seqs_padded, labels, lengths


def pad_sequence_denoising(batch):
    """
    Pad the sequence batch with 0 vectors.
    Compute the original lengths of each sequence.
    Collect labels into a dict of tensors.
    Meant to be used as the value of the `collate_fn`
    argument to `torch.utils.data.DataLoader`.
    """
    noisy_seqs = [torch.squeeze(x) for (x, _, _, _) in batch]
    noisy_seqs_padded = torch.nn.utils.rnn.pad_sequence(
            noisy_seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    seqs = [torch.squeeze(x) for (_, x, _, _) in batch]
    seqs_padded = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    lengths = torch.LongTensor([len(s) for s in seqs])
    labels = defaultdict(list)
    for (_, _, y, _) in batch:
        for label_name in y.keys():
            labels[label_name].append(y[label_name])
    for label_name in labels.keys():
        labels[label_name] = torch.stack(labels[label_name])
    ids = [i for (_, _, _, i) in batch]
    return noisy_seqs_padded, seqs_padded, labels, lengths, ids


# === RECONSTRUCT AND LOG INPUT ===
def tensor2text(tensor, idx2word, eos_token_idx):
    """
    Given a tensor of word indices, convert it to a list of strings.
    """
    try:
        eos = torch.where(tensor == eos_token_idx)[0][0]
    except IndexError:
        eos = tensor.size(0)
    return [idx2word[i.item()] for i in tensor[:eos+1]]


def get_reconstructions(model, examples, tokenizer):
    batch = min_pad_collate(examples)
    inX = send_to_device(batch["json"]["encoded"], model.device)
    output = model(inX, teacher_forcing_prob=0.0)

    recon_text = tokenizer.batch_decode(output["token_predictions"],
                                        skip_special_tokens=True)
    target_text = tokenizer.batch_decode(inX["input_ids"],
                                         skip_special_tokens=True)
    return [{"target": target_text[i], "reconstruction": recon_text[i]}
            for i in range(len(recon_text))]


def log_reconstructions(model, examples, tokenizer, name, epoch, logdir):
    # Log inputs and their reconstructions before model training
    recon_file = os.path.join(logdir, f"reconstructions_{name}.log")
    recons = get_reconstructions(model, examples, tokenizer)
    out_data = {"epoch": epoch, "reconstructions": recons}
    with open(recon_file, 'a') as outF:
        json.dump(out_data, outF)
        outF.write('\n')


def send_to_device(collection, device):
    if torch.is_tensor(collection):
        if collection.device != device:
            return collection.to(device)

    if isinstance(collection, dict):
        for key in collection.keys():
            collection[key] = send_to_device(collection[key], device)
    elif isinstance(collection, (list, tuple, set)):
        for i in range(len(collection)):
            collection[i] = send_to_device(collection[i], device)
    return collection

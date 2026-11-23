
import argparse
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


SUPPORTED_DIMS = {
    ('twitter', 'glove'): {25, 50, 100},
    ('muse', 'bpemb'): {50, 100},
    ('voxceleb2', 'wnta'): {64, 128},
}

BP_VOCAB_SIZES = {1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000}


def validate_config(dataset, embedding, dim, vocab_size=None, language=None):

    assert (dataset, embedding) in SUPPORTED_DIMS, f'Unsupported dataset/embedding: {dataset}/{embedding}'
    assert dim in SUPPORTED_DIMS[(dataset, embedding)], f'Available dimensions: {sorted(SUPPORTED_DIMS[(dataset, embedding)])}'

    if embedding == 'bpemb':
        assert vocab_size in BP_VOCAB_SIZES, f'Available vocabulary sizes: {sorted(BP_VOCAB_SIZES)}'
        assert language in {'en', 'en-es'}, 'MUSE language must be en or en-es'
        assert language != 'en-es' or dim == 100, 'Bilingual MUSE uses 100D in the paper'
    else:
        assert vocab_size is None, '--vocab-size is only applicable to BPEmb'
        assert language is None, '--language is only applicable to BPEmb'


def save_vectors(words, vectors, path):
    from gensim.models import KeyedVectors

    assert len(words) == len(vectors) and len(words) > 0

    model = KeyedVectors(vector_size=vectors.shape[1])
    model.add_vectors(words, np.asarray(vectors, dtype=np.float32))

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))

    print(f'Saved {len(words):,} vectors to {path}')
    return path


def get_muse_vocabulary(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    words = []

    with path.open(encoding='utf-8', errors='surrogateescape') as f:
        header = f.readline()
        parts  = header.split()

        if not (len(parts) == 2 and all(x.isdigit() for x in parts)):
            words.append(parts[0])

        for line in f:
            parts = line.split(maxsplit=1)
            if parts:
                words.append(parts[0])

    return words


def get_bpemb_vector(model, word, dim):

    value = np.asarray(model.embed(word))

    if value.shape == (1, dim) and np.isfinite(value).all():
        return value[0]

    return None


def get_muse_pairs(data_dir):

    source = get_muse_vocabulary(data_dir / 'muse/embeddings/wiki.multi.en.vec')
    target = set(get_muse_vocabulary(data_dir / 'muse/embeddings/wiki.multi.es.vec'))

    dictionary = data_dir / 'muse/dictionaries/en-es.txt'

    if not dictionary.is_file():
        raise FileNotFoundError(dictionary)

    valid_source = set(source)
    forward      = defaultdict(set)
    backward     = defaultdict(set)

    with dictionary.open(encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            words = line.split()

            if len(words) == 2 and words[0] in valid_source and words[1] in target:
                en, es = words
                forward[en].add(es)
                backward[es].add(en)

    pairs = []

    for en in source:
        if len(forward[en]) == 1:
            es = next(iter(forward[en]))

            if len(backward[es]) == 1:
                pairs.append((en, es))

    return pairs


def get_text_embeddings(dataset, embedding, dim, data_dir='../datasets', vocab_size=None, language=None, num_samples=400000, seed=42):

    if embedding == 'bpemb':
        vocab_size = 200000 if vocab_size is None else vocab_size
        language   = 'en' if language is None else language

    validate_config(dataset, embedding, dim, vocab_size, language)
    assert dataset in {'twitter', 'muse'}

    data_dir = Path(data_dir)

    ##################### TWITTER-GLOVE #####################

    if dataset == 'twitter':
        import gensim.downloader as api

        pretrained = api.load(f'glove-twitter-{dim}')
        vocabulary = sorted(pretrained.key_to_index)

        assert 0 < num_samples <= len(vocabulary)

        indices = sorted(random.Random(seed).sample(range(len(vocabulary)), num_samples))
        words   = [vocabulary[i] for i in indices]
        vectors = np.stack([pretrained[word] for word in tqdm(words, desc=f'Twitter GloVe {dim}D')])

        return save_vectors(words, vectors, data_dir / f'twitter_glove_{dim}.d2v')

    ##################### MUSE-BPEMB #####################

    from bpemb import BPEmb

    if language == 'en':
        words = get_muse_vocabulary(data_dir / 'muse/embeddings/wiki.multi.en.vec')
        model = BPEmb(lang='en', dim=dim, vs=vocab_size)

        retained = []
        vectors  = []

        for word in tqdm(words, desc=f'MUSE English BPEmb {dim}D'):
            value = get_bpemb_vector(model, word, dim)

            if value is not None:
                retained.append(word)
                vectors.append(value)

        return save_vectors(retained, np.asarray(vectors, dtype=np.float32), data_dir / f'muse_en_BP_{dim}_{vocab_size//1000}K.d2v')

    ##################### MUSE ENGLISH-SPANISH #####################

    pairs    = get_muse_pairs(data_dir)
    model_en = BPEmb(lang='en', dim=dim, vs=vocab_size)
    model_es = BPEmb(lang='es', dim=dim, vs=vocab_size)

    words_en   = []
    words_es   = []
    vectors_en = []
    vectors_es = []

    for en, es in tqdm(pairs, desc='MUSE English-Spanish'):
        x = get_bpemb_vector(model_en, en, dim)
        y = get_bpemb_vector(model_es, es, dim)

        if x is not None and y is not None:
            words_en.append(en)
            words_es.append(es)
            vectors_en.append(x)
            vectors_es.append(y)

    output_dir = data_dir / 'muse_bilingual_en_es'

    p_en = save_vectors(words_en, np.asarray(vectors_en, dtype=np.float32), output_dir / f'muse_en_BP_{dim}_{vocab_size//1000}K.d2v')
    p_es = save_vectors(words_es, np.asarray(vectors_es, dtype=np.float32), output_dir / f'muse_es_BP_{dim}_{vocab_size//1000}K.d2v')

    return p_en, p_es


def load_wnta(dim, device):

    import torch.nn as nn
    from transformers.models.wavlm.modeling_wavlm import WavLMPreTrainedModel, WavLMModel

    class TopLayers(nn.Module):

        def __init__(self, embd_size=250, top_interm_size=512):
            super().__init__()

            self.affine1    = nn.Conv1d(2048, top_interm_size, 1)
            self.batchnorm1 = nn.BatchNorm1d(top_interm_size, affine=False, eps=1e-3)
            self.affine2    = nn.Conv1d(top_interm_size, embd_size, 1)
            self.batchnorm2 = nn.BatchNorm1d(embd_size, affine=False, eps=1e-3)
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.batchnorm1(self.activation(self.affine1(x)))
            x = self.batchnorm2(self.activation(self.affine2(x)))
            return nn.functional.normalize(x[:, :, 0], dim=1)

    class EmbeddingsModel(WavLMPreTrainedModel):

        def __init__(self, config):
            super().__init__(config)

            self.wavlm      = WavLMModel(config)
            self.top_layers = TopLayers(config.embd_size, config.top_interm_size)

        def forward(self, input_values):
            x = (input_values - input_values.mean(dim=1, keepdim=True)) / input_values.std(dim=1, keepdim=True).clamp_min(1e-6)

            base  = self.wavlm(input_values=x, output_hidden_states=False).last_hidden_state
            stats = torch.cat((base.mean(dim=1), base.var(dim=1).clamp_min(1e-10).sqrt()), dim=1).unsqueeze(2)

            return self.top_layers(stats)

    return EmbeddingsModel.from_pretrained('Orange/Speaker-wavLM-pro', revision=f'wnta{dim}').to(device).eval()


def get_voice_embeddings(dataset, embedding, dim, data_dir='../datasets', audio_dir=None, num_files=600000, max_audio_samples=2000000, device='cuda'):

    validate_config(dataset, embedding, dim)
    assert dataset == 'voxceleb2'

    import torchaudio

    data_dir  = Path(data_dir)
    audio_dir = Path(audio_dir) if audio_dir else data_dir / 'voxceleba2/dev/aac'

    paths = sorted(audio_dir.rglob('*.m4a'))

    if not paths:
        paths = sorted(audio_dir.rglob('*.wav'))

    if not paths:
        raise FileNotFoundError(f'No audio files under {audio_dir}')

    assert 0 < num_files <= len(paths)

    paths  = paths[:num_files]
    device = torch.device(device)

    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    model = load_wnta(dim, device)
    data_dir.mkdir(parents=True, exist_ok=True)

    ##################### EXTRACT EMBEDDINGS #####################

    with tempfile.TemporaryDirectory(dir=data_dir) as folder:
        values = np.lib.format.open_memmap(Path(folder) / 'embeddings.npy', mode='w+', dtype=np.float32, shape=(len(paths), dim))
        count  = 0

        with torch.inference_mode():
            for path in tqdm(paths, desc=f'VoxCeleb2 WNTA {dim}D'):
                signal, sr = torchaudio.load(str(path))

                assert sr == 16000, f'Expected 16 kHz audio: {path}'

                signal = signal.mean(dim=0)

                if signal.numel() > max_audio_samples:
                    continue

                values[count] = model(signal.to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
                count += 1

        if not count:
            raise ValueError('No valid audio samples')

        tag    = f'{num_files//1000}K' if num_files % 1000 == 0 else str(num_files)
        output = data_dir / f'voxceleb2_wnta_dim_{dim}_{tag}.pt'

        torch.save(torch.from_numpy(np.array(values[:count])), output)

    print(f'Saved {count:,} voice embeddings to {output}')
    return output


def main():

    parser = argparse.ArgumentParser(description='Preprocess one GWBench embedding configuration.')

    parser.add_argument('--dataset', choices=['twitter', 'muse', 'voxceleb2'], required=True)
    parser.add_argument('--embedding', choices=['glove', 'bpemb', 'wnta'], required=True)
    parser.add_argument('--dim', type=int, required=True)

    parser.add_argument('--vocab-size', type=int, default=None)
    parser.add_argument('--language', choices=['en', 'en-es'], default=None)

    parser.add_argument('--data-dir', default='../datasets')
    parser.add_argument('--num-samples', type=int, default=400000)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--audio-dir', default=None)
    parser.add_argument('--num-files', type=int, default=600000)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    if args.embedding == 'bpemb':
        args.vocab_size = 200000 if args.vocab_size is None else args.vocab_size
        args.language   = 'en' if args.language is None else args.language

    validate_config(args.dataset, args.embedding, args.dim, args.vocab_size, args.language)

    if args.dataset == 'voxceleb2':
        get_voice_embeddings(args.dataset, args.embedding, args.dim, data_dir=args.data_dir, audio_dir=args.audio_dir, num_files=args.num_files, device=args.device)

    else:
        get_text_embeddings(args.dataset, args.embedding, args.dim, data_dir=args.data_dir, vocab_size=args.vocab_size, language=args.language, num_samples=args.num_samples, seed=args.seed)


if __name__ == '__main__':
    main()

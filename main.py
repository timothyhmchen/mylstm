"""Example script that trains an LSTM synonym translator with PyTorch."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


@dataclass
class Sample:
    """A single synonym mapping example."""

    source: str
    target: str
    site_id: str
    category_id: str


class Vocabulary:
    """Bidirectional mapping between string tokens and integer ids."""

    def __init__(self, tokens: Iterable[str]):
        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        unique_tokens = list(dict.fromkeys(specials + sorted(set(tokens))))
        self._index_to_token: List[str] = unique_tokens
        self._token_to_index: Dict[str, int] = {token: idx for idx, token in enumerate(unique_tokens)}
        self._unk_index = self._token_to_index[UNK_TOKEN]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._index_to_token)

    def token_to_index(self, token: str) -> int:
        return self._token_to_index.get(token, self._unk_index)

    def index_to_token(self, index: int) -> str:
        return self._index_to_token[index]


class LabelEncoder:
    """Utility for encoding categorical features such as site ids."""

    def __init__(self, labels: Iterable[str]):
        unique = sorted(set(labels))
        self._label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(unique)}
        self._index_to_label: List[str] = unique

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._index_to_label)

    def encode(self, label: str) -> int:
        return self._label_to_index[label]

    def decode(self, index: int) -> str:
        return self._index_to_label[index]


class SynonymDataset(Dataset[Tuple[List[int], List[int], int, int]]):
    """Dataset that stores word and feature indices."""

    def __init__(self, samples: Sequence[Sample], word_vocab: Vocabulary, site_encoder: LabelEncoder, category_encoder: LabelEncoder):
        self.samples = samples
        self.word_vocab = word_vocab
        self.site_encoder = site_encoder
        self.category_encoder = category_encoder

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _tokenise(self, text: str) -> List[str]:
        return text.lower().split()

    def _encode_words(self, text: str) -> List[int]:
        tokens = [SOS_TOKEN]
        tokens.extend(self._tokenise(text))
        tokens.append(EOS_TOKEN)
        return [self.word_vocab.token_to_index(token) for token in tokens]

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], int, int]:
        sample = self.samples[idx]
        source_ids = self._encode_words(sample.source)
        target_ids = self._encode_words(sample.target)
        site_id = self.site_encoder.encode(sample.site_id)
        category_id = self.category_encoder.encode(sample.category_id)
        return source_ids, target_ids, site_id, category_id


def collate_fn(batch: Sequence[Tuple[List[int], List[int], int, int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pad variable length sequences and convert them into tensors."""

    src_sequences, tgt_sequences, site_ids, category_ids = zip(*batch)

    src_lengths = torch.tensor([len(seq) for seq in src_sequences], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_sequences], dtype=torch.long)

    def pad(sequences: Sequence[List[int]]) -> Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), fill_value=0, dtype=torch.long)
        for idx, seq in enumerate(sequences):
            padded[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded

    src_batch = pad(src_sequences)
    tgt_batch = pad(tgt_sequences)
    site_tensor = torch.tensor(site_ids, dtype=torch.long)
    category_tensor = torch.tensor(category_ids, dtype=torch.long)

    return src_batch, src_lengths, tgt_batch, tgt_lengths, site_tensor, category_tensor


class SynonymTranslator(nn.Module):
    """Sequence-to-sequence model with feature-aware decoder initialisation."""

    def __init__(self, vocab_size: int, site_count: int, category_count: int, pad_idx: int, sos_idx: int, eos_idx: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.site_embeddings = nn.Embedding(site_count, embed_dim)
        self.category_embeddings = nn.Embedding(category_count, embed_dim)

        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.feature_to_hidden = nn.Linear(embed_dim * 2, hidden_dim)
        self.feature_to_cell = nn.Linear(embed_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def _initialise_decoder_state(self, encoder_state: Tuple[Tensor, Tensor], site_ids: Tensor, category_ids: Tensor) -> Tuple[Tensor, Tensor]:
        h_n, c_n = encoder_state
        site_embed = self.site_embeddings(site_ids)
        category_embed = self.category_embeddings(category_ids)
        features = torch.cat([site_embed, category_embed], dim=-1)

        feature_hidden = torch.tanh(self.feature_to_hidden(features)).unsqueeze(0)
        feature_cell = torch.tanh(self.feature_to_cell(features)).unsqueeze(0)

        return h_n + feature_hidden, c_n + feature_cell

    def forward(self, src: Tensor, src_lengths: Tensor, tgt: Tensor, site_ids: Tensor, category_ids: Tensor, teacher_forcing_ratio: float = 0.75) -> Tensor:
        embeddings = self.word_embeddings(src)
        packed = pack_padded_sequence(embeddings, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, encoder_state = self.encoder(packed)

        batch_size, tgt_seq_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_seq_len, self.output_layer.out_features, device=tgt.device)

        decoder_input = tgt[:, 0].unsqueeze(1)
        decoder_hidden = self._initialise_decoder_state(encoder_state, site_ids, category_ids)

        for t in range(1, tgt_seq_len):
            decoder_embedded = self.word_embeddings(decoder_input)
            decoder_output, decoder_hidden = self.decoder(decoder_embedded, decoder_hidden)
            step_logits = self.output_layer(decoder_output)
            outputs[:, t, :] = step_logits.squeeze(1)

            teacher_force = random.random() < teacher_forcing_ratio
            next_input = tgt[:, t] if teacher_force else step_logits.argmax(-1).squeeze(1)
            decoder_input = next_input.unsqueeze(1)

        return outputs

    def greedy_decode(self, src: Tensor, src_lengths: Tensor, site_ids: Tensor, category_ids: Tensor, max_len: int = 10) -> List[List[int]]:
        was_training = self.training
        self.eval()
        predictions: List[List[int]] = []
        with torch.no_grad():
            embeddings = self.word_embeddings(src)
            packed = pack_padded_sequence(embeddings, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, encoder_state = self.encoder(packed)

            decoder_input = torch.full((src.size(0), 1), fill_value=self.sos_idx, dtype=torch.long, device=src.device)
            decoder_hidden = self._initialise_decoder_state(encoder_state, site_ids, category_ids)

            sequences = [[] for _ in range(src.size(0))]
            for _ in range(max_len):
                decoder_embedded = self.word_embeddings(decoder_input)
                decoder_output, decoder_hidden = self.decoder(decoder_embedded, decoder_hidden)
                logits = self.output_layer(decoder_output)
                next_tokens = logits.argmax(-1)
                decoder_input = next_tokens

                all_eos = True
                for idx, token_id in enumerate(next_tokens.squeeze(1).tolist()):
                    sequences[idx].append(token_id)
                    if token_id != self.eos_idx:
                        all_eos = False
                if all_eos:
                    break
            predictions = sequences
        if was_training:
            self.train()
        return predictions


def build_samples() -> List[Sample]:
    """Create a small illustrative training corpus."""

    return [
        Sample("air jordan", "jordan", "nike_store", "sneakers"),
        Sample("air jordans", "jordan", "nike_store", "sneakers"),
        Sample("aj1", "jordan 1", "nike_store", "sneakers"),
        Sample("air max ninety", "air max 90", "nike_store", "sneakers"),
        Sample("yeezy boost", "yeezy", "adidas_store", "sneakers"),
        Sample("yeezy 350", "yeezy", "adidas_store", "sneakers"),
        Sample("ultraboost", "ultra boost", "adidas_store", "running"),
        Sample("ultra boost shoe", "ultra boost", "adidas_store", "running"),
        Sample("retro jordan", "jordan", "nike_store", "sneakers"),
        Sample("running ultra", "ultra boost", "adidas_store", "running"),
    ]


def build_vocabularies(samples: Sequence[Sample]) -> Tuple[Vocabulary, LabelEncoder, LabelEncoder]:
    word_tokens: List[str] = []
    site_ids: List[str] = []
    category_ids: List[str] = []
    for sample in samples:
        word_tokens.extend(sample.source.lower().split())
        word_tokens.extend(sample.target.lower().split())
        site_ids.append(sample.site_id)
        category_ids.append(sample.category_id)
    word_vocab = Vocabulary(word_tokens)
    site_encoder = LabelEncoder(site_ids)
    category_encoder = LabelEncoder(category_ids)
    return word_vocab, site_encoder, category_encoder


def translate(model: SynonymTranslator, text: str, site_id: str, category_id: str, dataset: SynonymDataset, word_vocab: Vocabulary, device: torch.device) -> str:
    model_input = dataset._encode_words(text)
    src = torch.tensor(model_input, dtype=torch.long, device=device).unsqueeze(0)
    src_lengths = torch.tensor([len(model_input)], dtype=torch.long, device=device)
    site_tensor = torch.tensor([dataset.site_encoder.encode(site_id)], dtype=torch.long, device=device)
    category_tensor = torch.tensor([dataset.category_encoder.encode(category_id)], dtype=torch.long, device=device)

    predicted = model.greedy_decode(src, src_lengths, site_tensor, category_tensor)[0]
    words: List[str] = []
    for token_id in predicted:
        if token_id in {word_vocab.token_to_index(PAD_TOKEN), word_vocab.token_to_index(EOS_TOKEN)}:
            break
        words.append(word_vocab.index_to_token(token_id))
    return " ".join(words)


def main() -> None:
    random.seed(7)
    torch.manual_seed(7)

    samples = build_samples()
    word_vocab, site_encoder, category_encoder = build_vocabularies(samples)

    dataset = SynonymDataset(samples, word_vocab, site_encoder, category_encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SynonymTranslator(
        vocab_size=len(word_vocab),
        site_count=len(site_encoder),
        category_count=len(category_encoder),
        pad_idx=word_vocab.token_to_index(PAD_TOKEN),
        sos_idx=word_vocab.token_to_index(SOS_TOKEN),
        eos_idx=word_vocab.token_to_index(EOS_TOKEN),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=word_vocab.token_to_index(PAD_TOKEN))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 300
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for src_batch, src_lengths, tgt_batch, _, site_tensor, category_tensor in dataloader:
            src_batch = src_batch.to(device)
            src_lengths = src_lengths.to(device)
            tgt_batch = tgt_batch.to(device)
            site_tensor = site_tensor.to(device)
            category_tensor = category_tensor.to(device)

            optimizer.zero_grad()
            outputs = model(src_batch, src_lengths, tgt_batch, site_tensor, category_tensor)
            logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
            targets = tgt_batch[:, 1:].reshape(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss / len(dataloader):.4f}")

    model.eval()
    print("\nSample translations:")
    examples = [
        ("air jordan", "nike_store", "sneakers"),
        ("ultraboost", "adidas_store", "running"),
        ("yeezy boost", "adidas_store", "sneakers"),
        ("retro jordan", "nike_store", "sneakers"),
    ]
    for text, site_id, category_id in examples:
        translated = translate(model, text, site_id, category_id, dataset, word_vocab, device)
        print(f"{text!r} -> {translated!r}")


if __name__ == "__main__":
    main()

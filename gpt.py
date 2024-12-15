import argparse
import os
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.3

# torch.manual_seed(1337)

# Dataset and encoding
def prepare_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, len(chars), encode, decode

train_data, val_data, vocab_size, encode, decode = prepare_data()

def get_batch(split, train_data, val_data, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Loss Estimation
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training function
def train_model(model, optimizer, train_data, val_data):
    best_val_loss = float('inf')  # Initialize with a large value
    best_model_path = "checkpoints/best_model.pth"

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4, verbose=True)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            # Evaluate losses on train and val sets
            losses = estimate_loss(model, train_data, val_data)
            train_loss, val_loss = losses['train'], losses['val']
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, perplexity {torch.exp(val_loss).item()}")

            # Save the model if the validation loss is the lowest so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with val loss {val_loss:.4f} at step {iter}")

            scheduler.step(val_loss)

        # Fetch a batch of training data
        xb, yb = get_batch('train', train_data, val_data, batch_size, block_size)

        # Forward pass and compute loss
        logits, loss = model(xb, yb)

        # Backward pass and optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Training complete. Best model saved at: {best_model_path} with val loss {best_val_loss:.4f}")

def generate_text(model, context, max_new_tokens):
    """
    Generate text from a trained model.

    Args:
        model: The trained GPT model.
        context: A tensor of shape (batch_size, context_length).
        max_new_tokens: Number of tokens to generate.

    Returns:
        Tensor of shape (batch_size, context_length + max_new_tokens).
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_cond = context[:, -block_size:]  # Truncate to block size (2D tensor)
            logits, _ = model(context_cond)  # Forward pass through the model
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            next_token = torch.multinomial(probs, num_samples=1)  # Sample the next token (2D: batch_size x 1)
            context = torch.cat((context, next_token), dim=1)  # Append the new token (2D tensor)
    return context

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "generate"], required=True, help="Mode to run the script in")
    parser.add_argument("--resume", action="store_true", help="Resume training from the best model checkpoint")
    parser.add_argument("--context", type=str, default="", help="Context string to generate text from")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of tokens to generate")
    args = parser.parse_args()

    # Initialize the model and optimizer
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if args.mode == "train":
        # Resume training if --resume is specified
        if args.resume:
            if os.path.exists("checkpoints/best_model.pth"):
                model.load_state_dict(torch.load("checkpoints/best_model.pth"))
                print("Resuming training from the best model checkpoint.")
            else:
                print("No checkpoint found to resume training. Starting from scratch.")
        train_model(model, optimizer, train_data, val_data)

    elif args.mode == "generate":
        # Load model for generation
        if os.path.exists("checkpoints/best_model.pth"):
            model.load_state_dict(torch.load("checkpoints/best_model.pth"))
            print("Loaded model from the best model checkpoint.")
        else:
            print("No trained model found. Please train the model first.")
            exit(1)

        # Handle empty context
        if not args.context:
            print("No context provided. Starting text generation from scratch.")
            # Initialize with a special start token or an empty tensor
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
        else:
            context = torch.tensor([encode(args.context)], dtype=torch.long, device=device)

        # Generate text
        generated = generate_text(model, context, args.max_new_tokens)
        print(decode(generated[0].tolist()))
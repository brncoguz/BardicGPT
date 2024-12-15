import unittest
import torch
from gpt import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel, get_batch, prepare_data

class TestGPTComponents(unittest.TestCase):
    def test_attention_head(self):
        """Test the output shape of a single attention head."""
        head = Head(head_size=64)
        x = torch.rand(2, 10, 384)  # (batch_size, seq_length, embedding_dim)
        output = head(x)
        self.assertEqual(output.shape, (2, 10, 64))

    def test_multi_head_attention(self):
        """Test the output shape of the multi-head attention module."""
        mha = MultiHeadAttention(num_heads=6, head_size=64)
        x = torch.rand(2, 10, 384)  # (batch_size, seq_length, embedding_dim)
        output = mha(x)
        self.assertEqual(output.shape, (2, 10, 384))  # Matches input embedding size

    def test_feed_forward(self):
        """Test the feedforward network's output shape."""
        ff = FeedFoward(n_embd=384)
        x = torch.rand(2, 10, 384)
        output = ff(x)
        self.assertEqual(output.shape, (2, 10, 384))  # Matches input shape

    def test_transformer_block(self):
        """Test a single transformer block's output shape."""
        block = Block(n_embd=384, n_head=6)
        x = torch.rand(2, 10, 384)
        output = block(x)
        self.assertEqual(output.shape, (2, 10, 384))  # Matches input shape

    def test_gpt_language_model_forward(self):
        """Test the forward pass of the GPTLanguageModel."""
        model = GPTLanguageModel()
        idx = torch.randint(0, 50, (2, 10))  # (batch_size, seq_length)
        logits, loss = model(idx)
        self.assertEqual(logits.shape, (2, 10, model.lm_head.out_features))  # Output logits shape
        self.assertIsNone(loss)  # Loss is None when targets are not provided

    def test_gpt_language_model_loss(self):
        """Test the GPT model's loss computation."""
        model = GPTLanguageModel()
        idx = torch.randint(0, 50, (2, 10))  # (batch_size, seq_length)
        targets = torch.randint(0, 50, (2, 10))
        logits, loss = model(idx, targets)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)

    def test_generate_function(self):
        """Test the generate function for output shape."""
        model = GPTLanguageModel()
        model.eval()  # Set the model to evaluation mode
        context = torch.randint(0, 50, (1, 5))  # (batch_size, context_length)
        generated = model.generate(context, max_new_tokens=10)
        self.assertEqual(generated.shape, (1, 15))  # context_length + max_new_tokens

    def test_get_batch(self):
        """Test the data batching function."""
        train_data = torch.arange(1000)
        val_data = torch.arange(1000, 1100)
        batch_size = 4
        block_size = 10

        x, y = get_batch('train', train_data, val_data, batch_size, block_size)
        self.assertEqual(x.shape, (batch_size, block_size))
        self.assertEqual(y.shape, (batch_size, block_size))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))  # Ensure x and y are aligned

    def test_prepare_data(self):
        """Test that data preparation returns the expected outputs."""
        train_data, val_data, vocab_size, encode, decode = prepare_data()
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)
        self.assertGreater(vocab_size, 0)

        # Encoding and decoding test
        sample_text = "hello"
        encoded = encode(sample_text)
        decoded = decode(encoded)
        self.assertEqual(sample_text, decoded)

    def test_block_gradient_flow(self):
        """Test that gradients flow correctly through a single block."""
        block = Block(n_embd=384, n_head=6)
        x = torch.rand(2, 10, 384, requires_grad=True)
        out = block(x)
        out.mean().backward()
        self.assertIsNotNone(x.grad)  # Ensure gradients are computed

    def test_model_gradient_flow(self):
        """Test that gradients flow correctly through the entire model."""
        model = GPTLanguageModel()
        idx = torch.randint(0, 50, (2, 10))  # (batch_size, seq_length)
        targets = torch.randint(0, 50, (2, 10))
        logits, loss = model(idx, targets)
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

if __name__ == "__main__":
    unittest.main()
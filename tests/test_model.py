"""Unit tests for ORBIT model."""

import unittest
import torch
from orbit.model import ORBIT


class TestORBIT(unittest.TestCase):
    """Test ORBIT model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ORBIT(
            speech_dim=1024,
            text_dim=768,
            hidden_dim=512,
            use_adversary=True,
            use_geometry=True,
        )
        self.batch_size = 4
        self.seq_len = 50

    def test_model_initialization(self):
        """Test model initializes."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.speech_dim, 1024)
        self.assertEqual(self.model.text_dim, 768)

    def test_forward_pass(self):
        """Test forward pass."""
        speech = torch.randn(self.batch_size, self.seq_len, 1024)
        text = torch.randn(self.batch_size, self.seq_len, 768)

        outputs = self.model(speech, text)

        self.assertIn('logits', outputs)
        self.assertIn('healthy_prob', outputs)
        self.assertIn('alzheimer_prob', outputs)
        self.assertEqual(outputs['logits'].shape[0], self.batch_size)

    def test_adversarial_mode(self):
        """Test adversarial computation."""
        speech = torch.randn(self.batch_size, self.seq_len, 1024)
        text = torch.randn(self.batch_size, self.seq_len, 768)

        outputs = self.model(speech, text, compute_adversarial=True)

        self.assertIn('adversarial_predictions', outputs)
        self.assertEqual(len(outputs['adversarial_predictions']), 3)  # 3 adversaries


if __name__ == '__main__':
    unittest.main()

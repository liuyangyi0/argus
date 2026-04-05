"""Tests for open vocabulary classifier (D1)."""

import numpy as np
import pytest

from argus.anomaly.classifier import OpenVocabClassifier, FOE_VOCAB


class TestOpenVocabClassifier:

    def test_classifier_init(self):
        """Classifier initializes with default vocabulary."""
        clf = OpenVocabClassifier()
        assert clf._vocabulary == FOE_VOCAB
        assert clf._loaded is False

    def test_custom_vocabulary(self):
        """Custom vocabulary is stored."""
        custom = ["cat", "dog", "bird"]
        clf = OpenVocabClassifier(vocabulary=custom)
        assert clf._vocabulary == custom

    def test_classify_returns_none_when_not_loaded(self):
        """Without model, classify returns None (graceful degradation)."""
        clf = OpenVocabClassifier(model_name="nonexistent.pt")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clf.classify(frame)
        # Should not crash, returns None since model can't load
        assert result is None

    def test_classify_empty_crop_returns_none(self):
        """Empty crop should return None."""
        clf = OpenVocabClassifier()
        clf._loaded = True  # Pretend loaded
        clf._model = None  # But no actual model
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = clf.classify(frame, bbox=(50, 50, 0, 0))
        assert result is None

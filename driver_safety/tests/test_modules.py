"""
test_modules.py — Unit Tests
Run: python -m pytest tests/ -v
"""
import sys, os, numpy as np, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES,
    VISION_WEIGHT, AUDIO_WEIGHT,
    FUSION_WARN_THRESH, FUSION_DROWSY_THRESH, FUSION_CRITICAL_THRESH,
    ALERT_LEVELS, LEFT_EYE_IDX, EAR_THRESHOLD, RANDOM_SEED
)


class TestConfig:
    def test_weights_sum_one(self):
        assert abs(VISION_WEIGHT + AUDIO_WEIGHT - 1.0) < 1e-6

    def test_thresholds_ordered(self):
        assert FUSION_WARN_THRESH < FUSION_DROWSY_THRESH < FUSION_CRITICAL_THRESH

    def test_alert_levels_complete(self):
        assert set(ALERT_LEVELS.keys()) == {0,1,2,3}

    def test_img_dims(self):
        assert IMG_HEIGHT == IMG_WIDTH == 64
        assert IMG_CHANNELS == 1


class TestPreprocess:
    def _mk_data(self, n=200):
        X = np.random.rand(n, IMG_HEIGHT, IMG_WIDTH, 1).astype(np.float32)
        y = np.array([0]*(n//2) + [1]*(n//2), dtype=np.int32)
        return X, y

    def test_balance(self):
        from vision.preprocess import balance_classes
        X = np.random.rand(150, 64, 64, 1).astype(np.float32)
        y = np.array([0]*100 + [1]*50, dtype=np.int32)
        Xb, yb = balance_classes(X, y)
        counts = np.bincount(yb)
        assert counts[0] == counts[1]

    def test_augment_shape(self):
        from vision.preprocess import augment_images
        X = np.random.rand(10, 64, 64, 1).astype(np.float32)
        y = np.array([0]*5+[1]*5, dtype=np.int32)
        Xa, ya = augment_images(X, y, factor=2)
        assert len(Xa) == 20
        assert len(ya) == 20

    def test_synthetic_dataset(self):
        from vision.preprocess import make_synthetic_dataset
        splits = make_synthetic_dataset(n=200)
        assert len(splits) == 6
        Xtr,Xv,Xte,ytr,yv,yte = splits
        assert Xtr.dtype == np.float32
        assert Xtr.max() <= 1.0


class TestCNNModel:
    def test_builds(self):
        from vision.cnn_model import build_cnn
        m = build_cnn()
        assert m.output_shape == (None, NUM_CLASSES)

    def test_feature_extractor(self):
        from vision.cnn_model import build_cnn
        fe = build_cnn(return_features=True)
        assert fe.output_shape == (None, 256)

    def test_forward_pass(self):
        from vision.cnn_model import build_cnn
        m = build_cnn()
        x = np.random.rand(4, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
        out = m.predict(x, verbose=0)
        assert out.shape == (4, NUM_CLASSES)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(4), atol=1e-5)

    def test_param_count(self):
        from vision.cnn_model import build_cnn
        m = build_cnn()
        n = m.count_params()
        assert 50_000 < n < 5_000_000, f"Unexpected param count {n}"


class TestLogisticRegression:
    def _fake_features(self, n=200, n_feat=256):
        np.random.seed(RANDOM_SEED)
        X = np.random.randn(n, n_feat).astype(np.float32)
        y = np.array([0]*(n//2)+[1]*(n//2), dtype=np.int32)
        return X, y

    def test_pipeline_builds(self):
        from vision.logistic_regression import build_lr_pipeline
        p = build_lr_pipeline()
        assert p is not None

    def test_fits_and_predicts(self):
        from vision.logistic_regression import build_lr_pipeline
        X, y = self._fake_features()
        p = build_lr_pipeline(use_pca=False)
        p.fit(X, y)
        preds = p.predict(X)
        assert len(preds) == len(y)
        proba = p.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_accuracy_above_chance(self):
        from vision.logistic_regression import build_lr_pipeline
        np.random.seed(RANDOM_SEED)
        # Clearly separable features
        X0 = np.random.randn(100, 50) + 3.0
        X1 = np.random.randn(100, 50) - 3.0
        X  = np.vstack([X0, X1]).astype(np.float32)
        y  = np.array([0]*100+[1]*100, dtype=np.int32)
        from sklearn.model_selection import train_test_split
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        p = build_lr_pipeline(use_pca=False)
        p.fit(Xtr, ytr)
        acc = (p.predict(Xte) == yte).mean()
        assert acc > 0.80, f"Expected >80% on separable data, got {acc:.3f}"


class TestEARDetector:
    def test_instantiates(self):
        from vision.ear_detector import EARDetector
        d = EARDetector()
        assert d is not None

    def test_blank_frame(self):
        from vision.ear_detector import EARDetector
        d = EARDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        r = d.process_frame(blank)
        assert r["face_detected"] == False
        assert r["alert_level"]   == 0

    def test_reset(self):
        from vision.ear_detector import EARDetector
        d = EARDetector()
        d.frame_counter = 20; d.total_blinks = 5
        d.reset()
        assert d.frame_counter == 0
        assert d.total_blinks  == 0


class TestMFCC:
    def _sig(self):
        sr = 22050; dur = 3.0
        t = np.linspace(0, dur, int(sr*dur))
        return (0.4*np.sin(2*np.pi*200*t) +
                0.1*np.random.randn(len(t))).astype(np.float32), sr

    def test_feature_shape(self):
        from audio.mfcc_extractor import extract_features
        y, sr = self._sig()
        f = extract_features(y, sr)
        assert f.ndim == 1 and len(f) > 100

    def test_from_array(self):
        from audio.mfcc_extractor import extract_from_array
        y, _ = self._sig()
        f = extract_from_array(y)
        assert f.ndim == 1

    def test_finite_values(self):
        from audio.mfcc_extractor import extract_features
        y, sr = self._sig()
        f = extract_features(y, sr)
        assert np.all(np.isfinite(f))


class TestFusionEngine:
    def _ear(self, perclos=0.05, ear=0.32, pitch=3.0):
        return dict(perclos=perclos, ear=ear, pitch=pitch, alert_level=0)

    def test_instantiates(self):
        from fusion.fusion_engine import FusionEngine
        assert FusionEngine() is not None

    def test_alert_level_0_when_normal(self):
        from fusion.fusion_engine import FusionEngine
        e = FusionEngine()
        for _ in range(12):
            r = e.update(cnn_drowsy_prob=0.05, ear_result=self._ear())
        assert r["alert_level"] == 0

    def test_level_rises_with_drowsiness(self):
        from fusion.fusion_engine import FusionEngine
        e = FusionEngine()
        ear = self._ear(perclos=0.45, ear=0.17, pitch=18)
        for _ in range(15):
            r = e.update(cnn_drowsy_prob=0.85, ear_result=ear)
        assert r["alert_level"] >= 2

    def test_critical_on_both_modalities(self):
        from fusion.fusion_engine import FusionEngine
        e = FusionEngine()
        ear = self._ear(perclos=0.5, ear=0.14, pitch=22)
        for _ in range(15):
            r = e.update(cnn_drowsy_prob=0.9, ear_result=ear,
                         audio_intox_prob=0.88)
        assert r["alert_level"] == 3

    def test_graceful_no_audio(self):
        from fusion.fusion_engine import FusionEngine
        e = FusionEngine()
        r = e.update(cnn_drowsy_prob=0.2, ear_result=self._ear(),
                     audio_intox_prob=None)
        assert r["modalities"]["audio_active"] == False
        assert r["fused_score"] >= 0

    def test_reset(self):
        from fusion.fusion_engine import FusionEngine
        e = FusionEngine()
        for _ in range(10):
            e.update(cnn_drowsy_prob=0.8, ear_result=self._ear(0.4,0.18))
        e.reset()
        assert e._alerts == 0 and len(e.history) == 0

    def test_session_summary(self):
        from fusion.fusion_engine import FusionEngine
        e = FusionEngine()
        for _ in range(20):
            e.update(cnn_drowsy_prob=0.1, ear_result=self._ear())
        s = e.session_summary()
        assert s["total_frames"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

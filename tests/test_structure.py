import json
from pathlib import Path


def test_available_topics_follow_the_learning_contract():
    topic_roots = [
        path.parent
        for path in Path("curriculum").glob("*/*/README.md")
        if path.parent.parent.name not in {"shared"}
    ]
    assert topic_roots
    for topic in topic_roots:
        notebooks = list(topic.glob("*.ipynb"))
        assert len(notebooks) == 1, topic
        assert (topic / "assets").is_dir(), topic
        assert (topic / "requirements.txt").is_file(), topic
        assert (topic / "constraints-tested.txt").is_file(), topic
        assert not list(topic.glob("*.py")), topic


def test_course_02_contains_the_declared_architecture_benchmark():
    course = Path("curriculum/beginner/02-modern-cnn-architectures-efficient-vision")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "ResNet-18",
        "ResNet-50",
        "MobileNetV3-Large",
        "EfficientNet-B0",
        "ConvNeXt-Tiny",
        "torch.profiler",
        "pareto_mask",
        "deployment_decision.json",
        "plain-deep",
        "residual-deep",
        "mac_reduction_vs_dense",
        "iqr_ms",
        "nearest_neighbor_retention",
        "CONTRACT_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "representation drift" in source_lower

    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_02_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/02-modern-cnn-architectures-efficient-vision/assets")
    expected = {
        "residual-block.svg",
        "efficient-convolution.svg",
        "cnn-family-evolution.svg",
        "systems-metrics.svg",
        "pareto-model-selection.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_course_03_contains_the_declared_transformer_benchmark():
    course = Path("curriculum/beginner/03-vision-transformers")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "patchify",
        "scaled_dot_product_attention",
        "PatchEmbed",
        "MultiHeadSelfAttention",
        "TransformerBlock",
        "TinyViT",
        "ResNet-50",
        "ConvNeXt-Tiny",
        "ViT-B/16",
        "Swin-T",
        "interpolate_embeddings",
        "small_defect_recall",
        "p95_ms_b1",
        "attention_distance",
        "attention_weight_memory",
        "global_vs_window_interactions",
        "flash_attention_compiled",
        "position_interpolation_summary",
        "gradient_times_input",
        "token_evolution",
        "transformer_decision.json",
        "CONTRACT_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "attention is not a causal explanation" in source_lower
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_03_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/03-vision-transformers/assets")
    expected = {
        "patch-token-pipeline.svg",
        "attention-qkv.svg",
        "transformer-encoder-block.svg",
        "vit-swin-hierarchy.svg",
        "cnn-vit-swin-comparison.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_course_04_contains_the_declared_ssl_learning_lab():
    course = Path("curriculum/beginner/04-self-supervised-visual-representation-learning")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "ntxent_loss",
        "temperature_results",
        "UnlabeledPairs",
        "domain_valid",
        "domain_invalid",
        "train_simclr",
        "teacher_student_loss",
        "distance_after_ema",
        "TinyMaskedAutoencoder",
        "masked-only MSE",
        "Supervised ImageNet · ResNet-18",
        "retrieval_precision_at_k",
        "collapse_diagnostics",
        "effective_rank",
        "class_minus_source",
        "LABEL_FRACTIONS",
        "patch_features",
        "CV_ENABLE_DINOV2",
        "representation_decision.json",
        "TEN_PERCENT_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "labels and source ids remain outside that training interface" in source_lower
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_04_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/04-self-supervised-visual-representation-learning/assets")
    expected = {
        "architecture-objective-shift.svg",
        "ssl-paradigms.svg",
        "contrastive-learning.svg",
        "teacher-student-learning.svg",
        "masked-image-modeling.svg",
        "representation-evaluation.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source

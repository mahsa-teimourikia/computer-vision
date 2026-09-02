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
        "train_teacher_student",
        "train_masked_reconstruction",
        "objective_history",
        "teacher_student_loss",
        "distance_after_ema",
        "TinyMaskedAutoencoder",
        "masked-only MSE",
        "Supervised ImageNet · ResNet-18",
        "retrieval_precision_at_k",
        "collapse_diagnostics",
        "collapse_comparison",
        "embedding_variance",
        "effective_rank",
        "top_singular_value_ratio",
        "Near-collapsed toy",
        "class_minus_source",
        "LABEL_FRACTIONS",
        "LABEL_SEEDS",
        "std_macro_f1",
        "patch_features",
        "horizontal_flip_correspondence",
        "patch_correspondence",
        "pretext_vs_downstream",
        "CV_ENABLE_DINOV2",
        "DINOV2_REVISION",
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "enterprise_options.csv",
        "enterprise_representation_decision.json",
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


def test_course_05_contains_the_declared_detection_lab():
    course = Path("curriculum/beginner/05-object-detection")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "xyxy_to_xywh",
        "box_iou",
        "detection_events",
        "precision_recall_ap",
        "generate_anchors",
        "TinyAnchorFreeDetector",
        "sigmoid_focal_loss",
        "local_xywh_to_global",
        "manual_nms",
        "torchvision_nms",
        "threshold_sweep",
        "source_size_evaluation",
        "ap50_ap75_example",
        "nms_crowding_example",
        "mitigation_comparison",
        "detection_error_events",
        "error_taxonomy_summary",
        "failure_slices",
        "linear_sum_assignment",
        "hungarian_cost",
        "hungarian_cost_decomposition",
        "classification_cost",
        "L1_box_cost",
        "GIoU_cost",
        "ULTRALYTICS_TESTED_VERSION",
        "CV_ENABLE_GROUNDING_DINO",
        "detector_decision.json",
        "DEMONSTRATION_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "factory c is never used for gradient updates or threshold selection" in source_lower
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_05_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/05-object-detection/assets")
    expected = {
        "detection-output-contract.svg",
        "anchor-vs-anchor-free.svg",
        "feature-pyramid.svg",
        "nms-duplicate-removal.svg",
        "dense-vs-set-prediction.svg",
        "detr-matching.svg",
        "detection-error-taxonomy.svg",
        "closed-vs-open-vocabulary.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_course_06_contains_the_declared_segmentation_lab():
    course = Path("curriculum/beginner/06-segmentation-promptable-segmentation")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "mask_quality_report",
        "bilinear: corrupted",
        "binary_iou",
        "dice_score",
        "pixel_accuracy",
        "per_class_iou",
        "mean_iou",
        "boundary_f1",
        "empty_policy_examples",
        "topology_signature",
        "TinyUNet",
        "ConvBlock",
        "DownBlock",
        "UpBlock",
        "multiclass_dice_loss",
        '"CE+Dice"',
        "slice_table",
        "point_prompt_proxy",
        "box_prompt_proxy",
        "prompt_sensitivity",
        "box_error_propagation",
        "LOCAL_PROMPT_ENGINE",
        '"foundation_model": False',
        "quality_rank_correlations",
        "quality_calibration_buckets",
        "correction_effort",
        "TARGET_REVIEW_IOU",
        "information_budget",
        "SAM31_REPO_REVISION",
        "SAM31_MODEL_REVISION",
        "daa63191845a41281374e725f4c9e51c7a824460",
        "record_sam31_quality_observation",
        "RUN_SAM31",
        "load_from_HF=False",
        "course-06-segmentation-evidence.json",
        "locally_measured_evidence",
        "optional_downloaded_model_observations",
        "unresolved_production_assumptions",
        "human_review_policy",
        "operating_contract_comparison_not_a_ranking",
    ]:
        assert required in source

    assert "local prompt proxy" in source_lower and "not a foundation model" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_06_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/06-segmentation-promptable-segmentation/assets")
    expected = {
        "segmentation-taxonomy.svg",
        "unet-encoder-decoder.svg",
        "semantic-vs-instance.svg",
        "mask-metrics.svg",
        "mask-rcnn.svg",
        "query-mask-classification.svg",
        "promptable-segmentation.svg",
        "detector-segmenter-pipeline.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_course_07_contains_the_declared_retrieval_lab():
    course = Path("curriculum/beginner/07-visual-embeddings-metric-learning-retrieval")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "contrastive_pair_loss",
        "triplet_loss_with_diagnostics",
        "make_pk_batch",
        "mine_online_triplets",
        "semi-hard",
        "batch-hard",
        "sampling_signal_table",
        "easy (farthest)",
        "mine_offline_triplets",
        "false_negative",
        "precision_at_k",
        "recall_at_k",
        "average_precision",
        "evaluate_retrieval",
        "ResNet18_Weights.DEFAULT",
        "TinyMetricEncoder",
        "region_image",
        "difference_hash",
        "hard_negative_review",
        "ambiguous taxonomy / representation mismatch",
        "IndexFlatIP",
        "IndexHNSWFlat",
        "FAISS_WORKER_SOURCE",
        "ann_recall_at_k",
        "ANN_Recall@10",
        "filtered_retrieval",
        "post-filter after top-5",
        "encoder_manifests",
        "cross_version_status",
        "neighbor_retention",
        "DINOV2_REPO_REVISION",
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "course-07-retrieval-evidence.json",
        "locally_measured_evidence",
        "optional_downloaded_model_observations",
        "unresolved_production_assumptions",
        "DEMONSTRATION_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "factory c never supplies training gradients" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_07_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/07-visual-embeddings-metric-learning-retrieval/assets")
    expected = {
        "embedding-space.svg",
        "siamese-network.svg",
        "triplet-learning.svg",
        "hard-negative-mining.svg",
        "retrieval-pipeline.svg",
        "exact-vs-ann.svg",
        "embedding-versioning.svg",
        "retrieval-failure-taxonomy.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source

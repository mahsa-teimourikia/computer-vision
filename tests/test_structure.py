import json
import tomllib
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
        "duplicate_group_split",
        "split_group_overlap",
        "cross_split_group_overlap",
        "hard_negative_review",
        "ambiguous taxonomy / representation mismatch",
        "simulated_reviewer_decision",
        "simulated hidden-label oracle",
        "IndexFlatIP",
        "IndexHNSWFlat",
        "FAISS_WORKER_SOURCE",
        "ann_recall_at_k",
        "ANN_Recall@10",
        "median_individual_query_ms",
        "p95_individual_query_ms",
        "individual_timing_samples",
        "filtered_retrieval",
        "post-filter after top-5",
        "shallow_post_ms",
        "mitigated_post_ms",
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
    assert "p95_ms_per_query" not in source
    assert 'duplicate_pairs["split"] = [' not in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "faiss-cpu==1.15.0" not in project["project"]["optional-dependencies"]["learner"]
    assert "faiss-cpu==1.15.0" in project["project"]["optional-dependencies"]["contributor"]


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


def test_course_08_contains_the_declared_tracking_pose_lab():
    course = Path("curriculum/beginner/08-tracking-keypoints-pose")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "make_timestamps",
        "ground_truth_id",
        "simulate_detections",
        "box_iou",
        "linear_sum_assignment",
        "hungarian_assignment",
        "class IoUTracker",
        "class StructuredTracker",
        "TrackState",
        "lifecycle_events",
        "scalar_recursive_filter_1d",
        "scalar position-only recursive filter; no velocity state",
        "geometry_only",
        "appearance_only",
        "combined",
        "byte_style",
        "iou_gate",
        "center_gate",
        "appearance_gate",
        "combined_cost_threshold",
        "secondary_combined_cost_threshold",
        "occlusion_slices",
        "slice_fragment_recoveries",
        "identity_consistency_f1_teaching",
        "hota_like_teaching_not_official",
        "export_motchallenge",
        "TRACKEVAL_REPO_REVISION",
        "12c8791b303e0a0b50f753af204249e622d0281a",
        "gaussian_heatmap",
        "decode_heatmap_argmax",
        "heatmap_resolution_comparison",
        "oks_like_teaching",
        "pose_geometry",
        "timestamp_velocity",
        "temporal_lag_frames",
        "track_pose_history",
        "failure_propagation",
        "natural_container_association_failure",
        "source_shift",
        "injected_pose_stage_noise_proxy",
        "pose_image_model_inference",
        "source_shift_experimental_boundary",
        "CV_ENABLE_BYTETRACK",
        "CV_ENABLE_MMPPOSE",
        "CV_ENABLE_TORCHVISION_KEYPOINT",
        "course-08-tracking-pose-evidence.json",
        "locally_measured_evidence",
        "optional_downloaded_model_observations",
        "unresolved_production_assumptions",
        "DEMONSTRATION_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "the tracker still never sees hidden identity" in source_lower
    assert "not official coco oks" in source_lower
    assert "gate = 1 - self.iou_threshold" not in source
    assert '"fragmentation_total": run["metrics"]["fragmentation"]' not in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_08_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/08-tracking-keypoints-pose/assets")
    expected = {
        "tracking-by-detection.svg",
        "track-lifecycle.svg",
        "association-cost.svg",
        "identity-switch.svg",
        "bytetrack-association.svg",
        "keypoint-heatmap.svg",
        "top-down-vs-bottom-up-pose.svg",
        "track-pose-state.svg",
        "tracking-failure-taxonomy.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source

    association_spec = json.loads((assets / "specs" / "association-cost.json").read_text(encoding="utf-8"))
    association_text = json.dumps(association_spec)
    for required in ["Hard gates", "Remaining pairs", "Weighted cost", "Hungarian match", "max_combined_cost"]:
        assert required in association_text


def test_course_09_contains_the_declared_foundation_vision_lab():
    course = Path("curriculum/beginner/09-vision-foundation-models-open-vocabulary")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "clip_symmetric_loss",
        "l2_normalize",
        "local_alignment_proxy",
        "PROMPT_TEMPLATES",
        "zero_shot_scores",
        "ensemble_text",
        "vocabulary_sensitivity",
        "vocabulary_boundary_example",
        "select_abstention_policy",
        "selected_on",
        "specialist_proxy",
        "self_supervised_proxy",
        "image_text_alignment_proxy",
        "evaluate_retrieval",
        "evaluate_source_bias",
        "retrieval_cross_source_results",
        "retrieval_source_bias_results",
        "same_label@5",
        "same_source@5",
        "source_bias_excess",
        "patch_features",
        "patch_correspondence_accuracy",
        "local_grounding_proxy",
        "parse_grounding_prompt",
        "relationship_features",
        "build_relation_graph",
        "red valve right of pipe",
        "category_detection",
        "attribute_grounding",
        "relational_phrase_grounding",
        "phrase_grounding",
        "box_prompt_segmenter_proxy",
        "oracle_box",
        "perturbed_box",
        "grounded_wrong_instance",
        '"foundation_model": False',
        "adaptation_results",
        "OPTIONAL_MODEL_MANIFESTS",
        "artifact_hash_status",
        "production_provenance_complete",
        "comparison_eligible",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "daa63191845a41281374e725f4c9e51c7a824460",
        "trust_remote_code=False",
        "course-09-foundation-vision-evidence.json",
        "locally_measured_evidence",
        "optional_downloaded_model_observations",
        "unresolved_production_assumptions",
        "DEMONSTRATION_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "siglip, or a foundation model" in source_lower
    assert "factory c remains untouched" in source_lower
    assert "factory c reporting only; no template selection or changes" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_course_09_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/beginner/09-vision-foundation-models-open-vocabulary/assets")
    expected = {
        "beginner-track-synthesis.svg",
        "foundation-capabilities.svg",
        "dual-encoder-alignment.svg",
        "global-patch-features.svg",
        "open-vocabulary-grounding.svg",
        "detector-segmenter-composition.svg",
        "adaptation-ladder.svg",
        "evaluation-contract.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_intermediate_01_contains_the_declared_vlm_lab():
    course = Path("curriculum/intermediate/01-vision-language-models")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "patch_features",
        "spatial_pool",
        "token_budget",
        "attention_and_kv_components",
        "LinearProjector",
        "MLPProjector",
        "QueryResampler",
        "CrossAttentionFusion",
        "TinyGenerativeVLM",
        "autoregressive",
        "Language-prior baseline",
        "Visual Evidence Ablation",
        "run_visual_evidence_suite",
        "expected_cf_flip_rate",
        "observed_cf_flip_rate",
        "counterfactual_pair_accuracy",
        "counterfactual_evidence_change_rate",
        "FROZEN_ABLATION_PROTOCOL",
        "counterfactual_diagnostic",
        "evidence_hit_rate",
        "SINGLE_PATCH_CAPABILITIES",
        "region_set",
        "relation_pair",
        "unsupported_by_single_patch_head",
        "P(evidence correct | answer correct)",
        "unanswerable_confidences",
        "selected_abstention_threshold",
        "FROZEN_ABSTENTION_POLICY",
        "selective_accuracy",
        "unanswerable_rejection_rate",
        "false_abstention_rate",
        "before_after_change_proxy",
        "validate_contract",
        "patch/box consistency",
        "mismatched patch and box",
        "schema-valid wrong evidence",
        "AutoProcessor",
        "AutoModelForImageTextToText",
        "apply_chat_template",
        "trust_remote_code=False",
        "7e3e67edbbed1bf9888184d9df282b700a323964",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        "artifact_hash_status",
        "production_provenance_complete",
        "comparison_eligible",
        "intermediate-01-vlm-evidence.json",
        "locally_measured_evidence",
        "optional_downloaded_model_observations",
        "unresolved_production_assumptions",
        "demonstration_thresholds_for_this_notebook_runtime_only",
    ]:
        assert required in source

    assert "local teaching vlm" in source_lower
    assert "factory c is test-only" in source_lower
    assert "not a foundation model" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_intermediate_01_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/intermediate/01-vision-language-models/assets")
    expected = {
        "dual-encoder-vs-generative-vlm.svg",
        "visual-token-interfaces.svg",
        "connector-patterns.svg",
        "fusion-taxonomy.svg",
        "multimodal-training-stages.svg",
        "resolution-token-budget.svg",
        "visual-evidence-ablation.svg",
        "vlm-evaluation-contract.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_intermediate_02_contains_the_declared_reasoning_lab():
    course = Path("curriculum/intermediate/02-multimodal-reasoning-verification")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "Evidence",
        "Claim",
        "ToolContract",
        "VerifierContract",
        "ToolEvent",
        "ReasoningNode",
        "topological_order",
        "execute_reasoning_graph",
        "verified_true",
        "verified_false",
        "uncertain",
        "unknown",
        "review_required",
        "spatial_relation",
        "count_objects",
        "less_than",
        "TOOL_CONTRACTS",
        "VERIFIER_CONTRACTS",
        "dependency_availability",
        "state_after_tool",
        "evidence_checks",
        "FROZEN_REVIEW_POLICY",
        "Factory B development only",
        "Factory C reporting only; no threshold or rule changes",
        "opaque_answer_proxy",
        "counterfactual_sensitivity_matrix",
        "unit_counterfactual_matrix",
        "unit_counterfactual_metrics",
        "development_counterfactual_summary",
        "held_out_counterfactual_summary",
        "relevant_expected_effect_accuracy",
        "relevant_no_flip_cases",
        "expected_effect_accuracy",
        "irrelevant_invariance",
        "contradiction_check",
        "inverse_consistency",
        "attribute_failure",
        "failure_taxonomy_summary",
        "tool_runtime_error",
        "runtime_failure_propagation",
        "tool_input_correct",
        "tool_execution_correct",
        "multi_image_result",
        "image_swap_detected",
        "OPTIONAL_MODEL_VERIFIER",
        "intermediate-02-multimodal-reasoning-evidence.json",
        "locally_measured_evidence",
        "optional_downloaded_model_observations",
        "unresolved_production_assumptions",
        "demonstration_thresholds_for_this_notebook_runtime_only",
        '"authorization": "none"',
        '"foundation_model": False',
    ]:
        assert required in source

    assert "local_structured_perception_proxy" in source
    assert "not a vlm, foundation model, or production perception benchmark" in source_lower
    assert "does not request, imitate, or store hidden chain-of-thought" in source_lower
    assert 'set(TOOL_CONTRACTS) == {"spatial_relation", "count_objects", "less_than"}' in source
    assert 'set(VERIFIER_CONTRACTS) == {"contradiction_check"}' in source
    assert 'runtime_failure_propagation["derived_claim_state"] == "unknown"' in source
    assert 'runtime_failure_propagation["terminal_decision"] == "review_required"' in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_intermediate_02_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/intermediate/02-multimodal-reasoning-verification/assets")
    expected = {
        "checked-reasoning-pipeline.svg",
        "reasoning-dependency-dag.svg",
        "evidence-contracts.svg",
        "deterministic-tool-boundary.svg",
        "counterfactual-verification.svg",
        "multi-image-binding.svg",
        "reasoning-failure-taxonomy.svg",
        "enterprise-reasoning-architecture.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_intermediate_03_contains_the_declared_document_intelligence_lab():
    course = Path("curriculum/intermediate/03-document-intelligence")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "PageContract",
        "RegionRecord",
        "pixel_to_normalized_box",
        "normalized_to_pixel_box",
        "pdf_points_to_pixels",
        "route_document_input",
        "local_ocr_proxy",
        "not an OCR benchmark",
        "character_error_rate",
        "word_error_rate",
        "confidence_reliability",
        "local_layout_proxy",
        "naive_yx_order",
        "column_aware_order",
        "pairwise_order_accuracy",
        "validate_table_schema",
        "merged_cell_correctness",
        "proximity_bind",
        "discover_key_candidates",
        "discover_value_candidates",
        "semantic_geometry_bind",
        "role_free_semantic_key_plus_geometry_v2",
        "ground_truth_role",
        "binding_summary",
        "proximity_failure",
        "TransformationRecord",
        "normalize_currency",
        "normalize_percentage",
        "normalize_date",
        "year_fmt",
        "expected_numeric_dates",
        "ambiguous_date_policy",
        "checkbox_proxy",
        "signature_authenticity",
        "detect_table_continuation",
        "detect_repeated_boilerplate",
        "bind_figures_to_captions",
        "resolve_field_candidates",
        "conflicting",
        "build_structured_document",
        "logical_document_sha256",
        "logical_document_hash",
        "page_render_sha256",
        "verify_field",
        "logical_document_hash_valid",
        "page_render_hash_matches",
        "source_identity_complete",
        "source_page_matches",
        "source_box_matches",
        "source_text_matches",
        "source_region_matches",
        "replay_transformation",
        "wrong_page",
        "wrong_logical_document_hash",
        "logical_document_content_tamper",
        "wrong_valid_box",
        "wrong_render_hash",
        "right_cell_wrong_page",
        "right_source_wrong_region",
        "wrong_cell",
        "unsupported_value",
        "bad_normalization",
        "provenance_failure_attribution",
        "Template B development only",
        "Template C reporting only; no threshold, parser, key alias, locale, or schema changes",
        "nonverified_field_rate",
        "perturbation_summary",
        "review_required",
        "OPTIONAL_TOOL_MANIFESTS",
        "CV_ENABLE_TESSERACT",
        "CV_ENABLE_TABLE_TRANSFORMER",
        "CV_ENABLE_PADDLEOCR_VL",
        "6951ffe10ce031374bcd04fe400811da1e7e04ad",
        "2357cbe2b5a5d1c03e54f32764f06058933b65ab",
        "7587a7ef111d9dcbf8ac695f1376ab7014340a0c",
        "c5630abae1d940eafe0697512a0325494b02ab42",
        "trust_remote_code=False",
        "readiness_checks",
        "missing_requirements",
        "license_review_approved",
        "intermediate-03-document-intelligence-evidence.json",
        "locally_measured_evidence",
        "optional_model_observations",
        "unresolved_production_assumptions",
        '"authorization": "none"',
    ]:
        assert required in source

    assert "document text is untrusted data" in source_lower
    assert "template c" in source_lower and "reporting only" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_intermediate_03_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/intermediate/03-document-intelligence/assets")
    expected = {
        "document-intelligence-pipeline.svg",
        "document-types.svg",
        "ocr-pipeline.svg",
        "reading-order.svg",
        "layout-taxonomy.svg",
        "table-structure.svg",
        "document-evidence-graph.svg",
        "enterprise-document-architecture.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_intermediate_04_contains_the_declared_multimodal_rag_lab():
    course = Path("curriculum/intermediate/04-multimodal-retrieval-rag")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "Principal",
        "EvidenceUnit",
        "RetrievalRequest",
        "RetrievalQuery",
        "RetrievalHit",
        "EvidenceBundle",
        "Claim",
        "canonical_source_id",
        "source_version",
        "indexed_source_version",
        "embedding_model_version",
        "chunker_version",
        "index_version",
        "reranker_version",
        "indexed_at",
        "effective_from",
        "effective_to",
        "public_request",
        "public_evidence_view",
        "EVALUATION_ONLY_FIELDS",
        "required_evidence_ids",
        "gold_answer",
        "evaluation_only_semantic_class",
        "Vendor A",
        "Vendor B",
        "Vendor C",
        "development_only",
        "reporting_only_no_changes",
        "authorized_units",
        "ALLOWED_FILTERS",
        "before scoring",
        "version_policy_filter",
        "current_source_versions",
        "current_only",
        "as_of",
        "historical_allowed",
        "bm25_scores",
        "semantic_vector",
        "visual_scores",
        "structured_scores",
        "late_interaction_score",
        "multi_vector_scores",
        "route_query",
        "decompose_query",
        "phase_visible_units",
        "generate_candidates",
        "reciprocal_rank_fusion",
        "deterministic_rerank",
        "assert_proxy_feature_contract",
        "PROXY_FEATURE_BUILDERS",
        "forbidden_parameters",
        "retrieval_metrics",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
        "complete_set",
        "representation_comparison",
        "image_to_corpus_retrieval",
        "source_shortcut_rate",
        "expand_hierarchy",
        "assemble_evidence",
        "freshness_check",
        "top_k_sweep",
        "retrieval_stage_waterfall",
        "Initial candidates",
        "After fusion",
        "After reranking",
        "After dedupe/hierarchy",
        "Final bundle",
        "complete_evidence_recall",
        "distractor_count",
        "deterministic_maximum",
        "local_generation_proxy",
        "not an LLM, VLM, foundation model, or generation-quality benchmark",
        "verify_claim_citations",
        "citation_metrics",
        "citation_failure_cases",
        "wrong_support",
        "outside_bundle",
        "unauthorized_citation",
        "wrong_value",
        "unsupported_claim",
        "oracle_bundle",
        "attribute_rag_failure",
        "retrieval_or_assembly",
        "generation_or_answer_policy",
        "unauthorized_retrieval_rate",
        "acl_principal_comparison",
        "policy-injection",
        "distractor_result",
        "counterfactual_update",
        "stale_excluded",
        "stale_excluded_before_scoring",
        "stale_never_ranked",
        "version_policy_table",
        "FROZEN_RETRIEVAL_POLICY",
        "Vendor B development only",
        "Vendor C reporting only; no route, weight, threshold, candidate depth, top-k, context budget, or rule changes",
        "safe_retrieval_log",
        "source_degradation",
        "evaluation_matrix",
        "classify_pipeline_failure",
        "routing_failure",
        "lexical_retrieval_failure",
        "dense_retrieval_failure",
        "fusion_failure",
        "reranking_failure",
        "missing_evidence",
        "distractor_contamination",
        "citation_failure",
        "generation_failure",
        "acl_failure",
        "stale_index_failure",
        "OPTIONAL_MODEL_MANIFESTS",
        "CV_ENABLE_FAISS",
        "CV_ENABLE_BGE_M3",
        "CV_ENABLE_SIGLIP2",
        "CV_ENABLE_RERANKER",
        "CV_ENABLE_COLQWEN",
        "CV_ENABLE_MULTIMODAL_RERANKER",
        "5617a9f61b028005a4858fdac845db406aefb181",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "92908120384b7a2110c5beda3ab29cbdb2c08e49",
        "0c630e356bbbf292ae0d8f050f54b8640f05919a",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        "trust_remote_code=False",
        "comparison_eligible",
        "intermediate-04-multimodal-rag-evidence.json",
        '"corpus_contract"',
        '"retrieval_units"',
        '"routing_metrics"',
        '"retrieval_metrics"',
        '"complete_evidence_recall"',
        '"fusion_metrics"',
        '"citation_metrics"',
        '"answer_metrics"',
        '"distractor_tests"',
        '"acl_tests"',
        '"freshness_tests"',
        '"version_policy_tests"',
        '"proxy_feature_contract"',
        '"evaluation_data_boundary"',
        '"retrieval_stage_waterfall"',
        '"failure_attribution"',
        "locally_measured_evidence",
        "optional_model_observations",
        "unresolved_production_assumptions",
        '"authorization": "none"',
        "DEMONSTRATION_THRESHOLD_NOTICE",
    ]:
        assert required in source

    assert "retrieved content is untrusted data" in source_lower
    assert "query text and retrieved content can never grant access" in source_lower
    assert "production embedding" in source_lower
    readme = (course / "README.md").read_text(encoding="utf-8")
    assert "typed version policy" in readme.lower()
    assert "before scoring" in readme.lower()
    assert "retrieval-stage waterfall" in readme.lower()
    assert "evaluation-only semantic classes" in readme.lower()
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_intermediate_04_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/intermediate/04-multimodal-retrieval-rag/assets")
    expected = {
        "multimodal-rag-pipeline.svg",
        "retrieval-units.svg",
        "multi-index-retrieval.svg",
        "hybrid-retrieval.svg",
        "evidence-assembly.svg",
        "citation-contract.svg",
        "acl-aware-retrieval.svg",
        "rag-failure-taxonomy.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_intermediate_05_contains_the_declared_video_language_lab():
    course = Path("curriculum/intermediate/05-video-language-understanding")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "VideoContract",
        "FrameObservation",
        "EventAnnotation",
        "ClipUnit",
        "VideoRequest",
        "VideoQuery",
        "TemporalEvidence",
        "TemporalClaim",
        "StreamingState",
        "presentation_timestamp",
        "variable_frame_rate",
        "Camera A",
        "Camera B",
        "Camera C",
        "development_only",
        "reporting_only_no_changes",
        "public_request",
        "EVALUATION_ONLY_FIELDS",
        "assert_proxy_gold_separation",
        "assert_public_frame_observations",
        "assert_deployable_temporal_boundary",
        "DEPLOYABLE_TEMPORAL_BOUNDARY",
        "uniform_sample",
        "random_sample",
        "event_aware_sample",
        "hierarchical_sample",
        "event_observation_recall",
        "Temporal aliasing",
        "bag_of_frames_representation",
        "order_aware_representation",
        "reverse_video_test",
        "temporal_dependence_teaching",
        "clip_features",
        "fixed_windows",
        "observable_change_segments",
        "oracle_event_segmentation_ceiling",
        "retrieve_clips",
        "authorized_clips",
        "candidate_retrieval_metrics",
        "event_recall_at_k",
        "complete_temporal_evidence_recall",
        "temporal_iou",
        "start_error_seconds",
        "end_error_seconds",
        "duration_error_seconds",
        "localize_from_observations",
        "interval_relation",
        "interval_duration",
        "build_temporal_event_graph",
        "ALLOWED_TEMPORAL_RELATIONS",
        "FORBIDDEN_CAUSAL_RELATIONS",
        "evidence_from_interval",
        "deduplicate_evidence",
        "complete temporal evidence",
        "answer_temporal_query",
        "stage_recall_report",
        "stagewise_temporal_recall",
        "TEMPORAL_STAGE_ORDER",
        "raw_public_observations",
        "observable_after_sampling",
        "localized_intervals",
        "after_deduplication",
        "final_evidence_bundle",
        "temporal_failure_attribution",
        "verify_temporal_claim",
        "citation_not_in_bundle",
        "citation_not_authorized",
        "track_binding_missing",
        "unsupported_temporal_relation",
        "causal_relation_requires_separate_contract",
        "causal_overreach",
        "hallucination_fixtures",
        "Evidence ablation",
        "relevant_counterfactual",
        "irrelevant_counterfactual",
        "hierarchy_cost",
        "multi_scale_retrieve",
        "temporal_index_record",
        "evidence_bound_summary",
        "language_prior_claim",
        "stream_event_state",
        "possible_onset",
        "completion_delay_seconds",
        "simulate_backpressure",
        "BACKPRESSURE_SEEDS",
        "backpressure_trials",
        "event_aware_drop",
        "frame_drop_report",
        "FRAME_DROP_SEEDS",
        "random_delivery",
        "frame_drop_trials",
        "short_event_recall",
        "long_event_recall",
        "audio_visual_alarm_check",
        "conflict_abstain",
        "FROZEN_POLICY",
        "POLICY_HASH",
        "source_report",
        "OPTIONAL_ADAPTERS",
        "0d24878d4107d64bef49e53602fc34ce6f94f6d8",
        "488eb9a0565f257b32866000305c8178965eb9f6",
        "89644892e4d85e24eaac8bacfd4f463576704203",
        "trust_remote_code",
        "safe_observability_record",
        "local_temporal_representation_proxy",
        "local_video_language_proxy",
        '"foundation_model": False',
        '"authorization": "none"',
        "intermediate-05-video-language-evidence.json",
        "Demonstration thresholds for this notebook runtime only.",
    ]:
        assert required in source

    assert "not production video encoders" in source_lower
    assert "camera c / reporting only" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_intermediate_05_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/intermediate/05-video-language-understanding/assets")
    expected = {
        "video-language-pipeline.svg",
        "frame-index-vs-timestamp.svg",
        "temporal-sampling-aliasing.svg",
        "space-time-representations.svg",
        "temporal-retrieval-hierarchy.svg",
        "temporal-event-graph.svg",
        "temporal-citation-contract.svg",
        "offline-vs-streaming.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source


def test_intermediate_06_contains_the_declared_bounded_visual_agent_lab():
    course = Path("curriculum/intermediate/06-visual-agents")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "Principal",
        "Evidence",
        "VerifiedFact",
        "BudgetPolicy",
        "BudgetUsage",
        "ToolContract",
        "ProposedAction",
        "ToolResult",
        "TraceEvent",
        "AgentState",
        "TaskCase",
        "Site A",
        "Site B",
        "Site C",
        "construction",
        "development_only",
        "reporting_only_no_changes",
        "PRIVATE_EVALUATION_TRUTH",
        "TOOL_REGISTRY",
        "ALLOWED_SIDE_EFFECTS",
        "READ_ONLY",
        "LOCAL_TRANSFORM",
        "detect_components",
        "inspect_image",
        "segment_component",
        "count_objects",
        "search_manual",
        "retrieve_video_event",
        "temporal_relation",
        "calculate",
        "apply_maintenance_rule",
        "validate_tool_registry",
        "authorize",
        "PERMISSION_POLICY",
        "validate_arguments",
        "box_out_of_bounds",
        "validate_result",
        "stale_source_version",
        "local_planner_proxy",
        "progress_hash",
        "execute_action",
        "permission_denied",
        "run_agent",
        "loop_detected",
        "goal_achieved",
        "evidence_sufficient",
        "review_required",
        "direct_answer_baseline",
        "verify_claim_support",
        "hallucinated_tool_result",
        "tool_selection_failure",
        "tool_input_failure",
        "tool_runtime_failure",
        "tool_output_failure",
        "stale_observation",
        "unauthorized_executions",
        "required_tool_recall",
        "unnecessary_tool_rate",
        "argument_validity",
        "verification_coverage",
        "transient timeout then recovery",
        "viewer permission",
        "video tool removed",
        "visual_text_trust",
        "document_text_trust",
        "relevant_counterfactual_tools",
        "irrelevant_metadata_plan_stable",
        "FROZEN_POLICY",
        "FROZEN_POLICY_HASH",
        "OPTIONAL_INTEGRATIONS",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "trust_remote_code",
        "proposal_only",
        "langgraph",
        "openai_agents_sdk",
        "mcp_gateway",
        "validate_optional_proposal",
        "intermediate-06-visual-agent-evidence.json",
        '"foundation_model": False',
        '"authorization": "none"',
    ]:
        assert required in source

    assert "not a foundation-agent reasoning benchmark" in source_lower
    assert "no shell, network, filesystem mutation" in source_lower
    assert "task success cannot compensate" in source_lower
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_advanced_01_contains_the_declared_spatial_intelligence_lab():
    course = Path("curriculum/advanced/01-3d-vision-spatial-intelligence")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "FramedPoints",
        "CameraModel",
        "make_transform",
        "invert_rigid",
        "transform_points",
        "project_points",
        "pixel_ray",
        "backproject_z",
        "distort_normalized",
        "estimate_camera_dlt",
        "error_summary",
        "fundamental_from_cameras",
        "sampson_error",
        "eight_point_F",
        "ransac_fundamental",
        "disparity_to_z",
        "stereo_sigma_z",
        "triangulate_dlt",
        "triangulation_monte_carlo",
        "align_scale_shift",
        "depth_metrics",
        "edge_RMSE_m",
        "pose_residual",
        "rotation_error_deg",
        "point_bundle_residual",
        "cloud_metrics",
        "chamfer_m",
        "voxelize",
        "fit_plane_svd",
        "zbuffer",
        "clearance_trials",
        "DEMONSTRATION_CLEARANCE_LIMIT_M",
        "Site A",
        "Site B",
        "Site C",
        "development_only",
        "reporting_only_no_changes",
        "FROZEN_POLICY",
        "POLICY_HASH",
        "evaluate_source",
        "CV_ENABLE_DEPTH_ANYTHING_V2",
        "CV_ENABLE_UNIDEPTH",
        "CV_ENABLE_VGGT",
        "CV_ENABLE_VGGT_OMEGA",
        "trust_remote_code",
        "spatial_intelligence_evidence.json",
        "spatial_tool_decision.csv",
    ]:
        assert required in source

    assert "relative aligned for shape only" in source_lower
    assert "occluded/unknown, not free space" in source_lower
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_advanced_01_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/01-3d-vision-spatial-intelligence/assets")
    expected = {
        "coordinate-frames.svg",
        "pinhole-projection.svg",
        "epipolar-geometry.svg",
        "stereo-depth.svg",
        "triangulation-uncertainty.svg",
        "sfm-pipeline.svg",
        "representation-landscape.svg",
        "spatial-evidence-pipeline.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source
    for spec in (assets / "specs").glob("*.json"):
        data = json.loads(spec.read_text(encoding="utf-8"))
        assert data["validation"]["status"] == "validated"
        assert "20px_clearance" in data["validation"]["checked"]
        assert data["source"]["deterministic"] is True


def test_intermediate_06_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/intermediate/06-visual-agents/assets")
    expected = {
        "visual-agent-loop.svg",
        "agent-state.svg",
        "tool-gateway.svg",
        "planning-vs-execution.svg",
        "verification-loop.svg",
        "agent-failure-taxonomy.svg",
        "cross-modal-agent.svg",
        "human-approval-boundary.svg",
    }
    assert {path.name for path in assets.glob("*.svg")} == expected
    assert {path.name for path in (assets / "specs").glob("*.json")} == {
        name.replace(".svg", ".json") for name in expected
    }
    for svg in assets.glob("*.svg"):
        source = svg.read_text(encoding="utf-8")
        assert "<title" in source and "<desc" in source
        assert 'role="img"' in source

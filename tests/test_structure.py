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
        "RigidTransform",
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
        "relative_depth_metrics",
        "metric_depth_metrics",
        "spearman_rank_correlation",
        "pairwise_ordering_accuracy",
        "oracle_aligned_shape_RMSE_m",
        "edge_RMSE_m",
        "pose_residual",
        "rotation_error_deg",
        "point_bundle_residual",
        "cloud_metrics",
        "symmetric_mean_nn_distance_m",
        "voxelize",
        "fit_plane_svd",
        "zbuffer",
        "clearance_trials",
        "teaching_interval_under_point_noise_model",
        "calibration covariance",
        "systematic scale error",
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

    assert "1000 mm points plus a metre transform must be rejected" in source
    assert '"prediction": "relative raw (not metres)", **depth_metrics' not in source
    assert '"chamfer_m"' not in source

    assert "shape evaluation after ground-truth alignment; not metric zero-shot accuracy" in source_lower
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


def test_advanced_02_contains_the_declared_neural_rendering_lab():
    course = Path("curriculum/advanced/02-neural-rendering-3d-scene-representations")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "CameraModel",
        "RayBundle",
        "generate_rays",
        "ray_distance_to_camera_z",
        "depth_conversion_report",
        "ray_box_intersections",
        "positional_encoding",
        "volume_render",
        "min_opacity_for_ray_distance",
        "expected_ray_distance_m",
        "scalar_loss_and_gradient",
        "numeric=",
        "coarse-guided fine samples",
        "occupancy",
        "ViewRecord",
        "Site A",
        "Site B",
        "Site C",
        "development_only",
        "reporting_only_no_changes",
        "same RGB, incompatible depth",
        "2.0",
        "4.0",
        "global_ssim_teaching",
        "GaussianPrimitive",
        "project_gaussian",
        "splat_render",
        "expected_camera_z_m",
        '"depth_convention":"camera_axis_z"',
        "sh_degree1_teaching",
        "apply_density_control",
        "split_child_opacity",
        '"global_opacity_reset_implemented":False',
        "clone",
        "split",
        "prune",
        "retain",
        "gaussian_artifact_bytes",
        "p90_ms",
        "DEMONSTRATION_POLICY",
        "FROZEN_POLICY_HASH",
        "camera_z_rmse",
        "np.sqrt(np.mean",
        "surface_support_coverage",
        "joint_valid_pixels",
        "reference_surface_pixels",
        "camera_z_RMSE_m",
        "support_coverage",
        "heldout_representation_comparison",
        "ablation_bad_z",
        "B_wrong_geometry_same_RGB",
        '"appearance_gate"',
        '"geometry_gate"',
        '"support_gate"',
        "CV_ENABLE_NERFSTUDIO=False",
        "CV_ENABLE_GSPLAT=False",
        "CV_ENABLE_INSTANT_NGP=False",
        "CV_ENABLE_ORIGINAL_3DGS=False",
        "CV_ENABLE_PYTORCH3D=False",
        "OPTIONAL_INTEGRATIONS",
        "abe236ee00cf90cfca6e36e65c00435d5",
        "e4f1b665d6ba51978ac786a313921cc1cfc9f293",
        "9bd7153cc96e6e85398bb8e70b382817149c1f03",
        "a3063beb58cb6d2943ecac68993bd54b483a81c8",
        "bf985a9eab3d0f127dc2d6043792f967739131e",
        "neural_rendering_evidence.json",
        "scene_representation_decision.csv",
    ]:
        assert required in source

    assert "not standard ssim" in source_lower
    assert "not nerf or 3dgs benchmark" in source_lower
    assert "appearance gain cannot compensate" in source_lower
    assert "mean-depth sorting is a teaching approximation" in source_lower
    assert '"expected_depth_m"' not in source
    assert '"depth_RMSE_m"' not in source
    assert "depth_error_m=" not in source
    assert "opacity=max(" not in source
    assert "opacity_reset_to" not in source
    assert "camera_depth_m" not in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_advanced_02_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/02-neural-rendering-3d-scene-representations/assets")
    expected = {
        "representation-taxonomy.svg",
        "differentiable-rendering-loop.svg",
        "volume-rendering-ray.svg",
        "camera-split.svg",
        "geometry-appearance-disagreement.svg",
        "nerf-vs-gaussian.svg",
        "gaussian-projection.svg",
        "gaussian-density-control.svg",
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


def test_advanced_03_contains_the_declared_world_model_lab():
    course = Path("curriculum/advanced/03-dynamic-scenes-world-models")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "ObjectState",
        "WorldState",
        "Action",
        "Observation",
        "HiddenDynamics",
        "step_environment",
        "render_observation",
        "same observation does not imply same predictive state",
        "PersistentStateEstimator",
        "evaluation_only_next_state",
        "development_only",
        "reporting_only_no_changes",
        "LocalWorldModelProxy",
        "local_world_model_proxy",
        "foundation_model",
        "Ridge",
        "action_conditioned",
        "action_ignorant",
        "evaluate_one_step",
        "evaluate_rollout",
        "open_loop_recursive",
        "position_vector_RMSE_m",
        "velocity_vector_RMSE_mps",
        "valve_state_accuracy",
        "VALVE_ACTIONS",
        "MODEL_ACTIONS",
        "valve_transition_report",
        "Valve actions must target valve_1",
        "previous_timestamp_s",
        "Observation timestamps must increase for each known object",
        "elapsed_time_velocity_test",
        "known_identity_persistence_report",
        "action_sensitivity_teaching",
        "action_consistency_rate",
        "wrong_action_test",
        "counterfactual_matrix",
        "valid_mode_coverage",
        "invalid_future_rate",
        "mode_frequency_TV_distance",
        "known_answer_metrics",
        "NOISY_MODEL_PROBABILITIES",
        "time_to_event_metrics",
        "timing_MAE_s",
        "timing_p95_abs_s",
        "validate_rollout",
        "planning_model_step",
        "true_plan_step",
        "planning_model_exploitation",
        "support_aware_score",
        "FROZEN_POLICY_HASH",
        "policy_hash_before_site_c",
        "policy_hash_after_site_c",
        "failure_attribution",
        "CV_ENABLE_DREAMERV3=False",
        "CV_ENABLE_COSMOS3=False",
        "CV_ENABLE_DYNAMIC_3D_GAUSSIANS=False",
        "e3f02248693a79dc8b0ebd62c93683888ddaccfe",
        "5a68d9d4d34c9ca2bdcb0a1d9bbbb3a2d1b8d497",
        "7dbbd4dec404308524ff402756bdb8143a2589b0",
        "world_model_evidence.json",
        "world_model_decision.csv",
        '"authorization": "none"',
    ]:
        assert required in source

    assert "planner never queries evaluation truth while selecting" in source_lower
    assert "not a foundation world model" in source_lower
    assert "generated video can never authorize an actuator" in source_lower
    assert "def predict(self, state: WorldState, action: Action)" in source
    assert "def predict(self, state: WorldState, action: Action, hidden" not in source
    assert "observation.timestamp_s - previous_timestamp_s" in source
    assert "observation.timestamp_s - 0.0" not in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_advanced_03_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/03-dynamic-scenes-world-models/assets")
    expected = {
        "world-model-loop.svg",
        "observation-vs-state.svg",
        "world-model-taxonomy.svg",
        "dynamic-scene-representations.svg",
        "rollout-error.svg",
        "counterfactual-actions.svg",
        "world-model-planning.svg",
        "model-exploitation.svg",
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


def test_advanced_04_contains_the_declared_embodied_vla_lab():
    course = Path("curriculum/advanced/04-embodied-vision-vla-models")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "EmbodimentContract",
        "SceneObject",
        "Observation",
        "GroundedGoal",
        "ActionProposal",
        "Action without frame is invalid",
        "source_capture_timestamp_s",
        'translation_unit: Literal["metre"]',
        "CAMERA_TO_BASE",
        "LocalGroundingProxy",
        "clarification_required",
        "unsafe_action_after_ambiguity_rate",
        "AffordanceRecord",
        "grasp_affordance",
        "ActionNormalizer",
        "normalize",
        "denormalize",
        "tokenize_action",
        "tokenization_translation_vector_error_m",
        "Ridge",
        "vision_only_policy",
        "proprioception_aware_policy",
        "translation_vector_MAE_m",
        "rollout_bc",
        "LocalVLAPolicyProxy",
        "local_vla_policy_proxy",
        "foundation_model = False",
        "ValidationDecision",
        "validate_action",
        "stale_observation",
        "embodiment_mismatch",
        "delta_limit_exceeded",
        "unreachable_target",
        "collision_proxy_violation",
        "gripper_incompatible",
        "unsafe_high_confidence",
        "plan_chunk",
        "open_loop_chunk",
        "receding_horizon",
        "visual_servo",
        "control_frequency_hz",
        "SimulationPermit",
        "simulation_only",
        "SimulationExecutor",
        "consumed_nonces",
        "replay_blocked",
        "mutated_proposal_blocked",
        "PostconditionEvidence",
        "verify_grasp",
        "verify_release",
        "episode_postconditions",
        "result_block_observed",
        "propose_pick_place_step",
        "closed_loop_task_metrics",
        "closed_loop_pick_place_success",
        "steps_to_completion",
        "proposal_violation_rate",
        "executed_violation_rate",
        "valid_work_block_rate",
        "FROZEN_POLICY_HASH",
        "policy_hash_before_site_c",
        "policy_hash_after_site_c",
        "affordance_compatible_rate_arm_B",
        "combined_feasible_action_rate_arm_B",
        "reporting only; no model, feature, normalizer, threshold, chunk, or rule changes",
        "failure_attribution",
        "OPTIONAL_TOOL_MANIFESTS",
        "CV_ENABLE_LEROBOT=False",
        "CV_ENABLE_OPENVLA=False",
        "CV_ENABLE_OPENPI=False",
        "CV_ENABLE_GR00T=False",
        "CV_ENABLE_MANISKILL=False",
        "CV_ENABLE_MUJOCO=False",
        "5aa74557f84c54d4b458f8b9643c5aa2982acfed",
        "c8f03f48af692657d3060c19588038c7220e9af9",
        "215abfb217dbac7d5f1273282331b9b1866c0479",
        "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
        "embodied_vla_evidence.json",
        "embodied_vla_decision.csv",
        '"physical_authorization": "none"',
        '"authorization": "none"',
    ]:
        assert required in source

    assert "not a vlm, foundation model, or grounding benchmark" in source_lower
    assert "no cell connects to hardware" in source_lower
    assert "policy output is untrusted data" in source_lower
    assert "sensor_age_s = now_s - observation.camera_captured_at_s" in source
    assert '["feedback", "recovery_attempt_rate", recovery_attempt_rate]' in source
    assert 'if post.task_effect_verified:\n            held, ee, block = True' in source
    assert 'if post.task_effect_verified:\n            held, ee, block, task_success = False' in source
    assert 'held = True\n' not in source
    assert 'task_success = True\n' not in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_advanced_04_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/04-embodied-vision-vla-models/assets")
    expected = {
        "embodied-closed-loop.svg",
        "model-role-taxonomy.svg",
        "embodiment-contract.svg",
        "grounding-affordance.svg",
        "action-space-taxonomy.svg",
        "action-chunking-feedback.svg",
        "action-verification-gateway.svg",
        "stale-observation.svg",
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


def test_advanced_05_contains_the_declared_spatial_memory_lab():
    course = Path("curriculum/advanced/05-spatial-memory-scene-graphs-navigation")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "Pose2D",
        "TrueWorld",
        "SensorPacket",
        "sensor_proxy",
        "trajectory_metrics",
        "ATE_RMSE_m",
        "RPE_RMSE_m",
        "OccupancyMemory",
        "log_odds",
        "unknown",
        "ObjectObservation",
        "MemoryObject",
        "association_score",
        "duplicate landmark",
        "identity merge risk",
        "FROZEN_ASSOCIATION",
        "apply_negative_evidence",
        "location_history",
        "location_at",
        "RelationRecord",
        "INVERSE",
        "TRANSITIVE",
        "validate_relation",
        "missing_provenance",
        "SpatialQueryEngine",
        "current_query",
        "historical_query",
        "local_semantic_retrieval_proxy",
        "candidate_only",
        "astar",
        "known_path",
        "unknown_policy",
        "inflate_obstacles",
        "topological_route",
        "NavigationPlan",
        "PlanDependencies",
        "resource_keys",
        "invalidate_plan_if_affected",
        "stale_memory_failure",
        "memory_assisted_recovery",
        "memoryless_search",
        "naive_stale_memory",
        "freshness_aware_memory",
        "strategy_comparison",
        "wrong_location_visits",
        "successful_recovery",
        "replanned_path",
        "loop_candidates",
        "TrustedMapState",
        "LoopClosureCandidate",
        "canonical_map_state",
        "propose_loop_closure",
        "verify_loop_closure",
        "apply_verified_loop_closure",
        "before_rejected_loop",
        "rejected_route_before",
        "rejected_route_after",
        "loop_non_mutation_report",
        "verified_false_loop_rate",
        "viewpoint_memory",
        "active_view",
        "FROZEN_CONFIG",
        "policy_hash_before_site_c",
        "policy_hash_after_site_c",
        "reporting_only_no_changes",
        "site_c_metrics",
        "evaluation_only_site_c",
        "site_c_packets",
        "failure_attribution",
        "TrustedMemoryManager",
        "corrupt_edge",
        "poison_candidate",
        "memory_update_trace",
        "affected_resources",
        "unrelated_update_report",
        "relevant_update_report",
        "plan_invalidation_comparison",
        "OPTIONAL_TOOL_MANIFESTS",
        "CV_ENABLE_HYDRA=False",
        "2e58a35baea629eee8838409771876acff015daf",
        "0fb6f43ffe806a8088a171b036336c093bcf604e",
        "93277a02bd89171f8121e84203121cf7af9ebb5d",
        "f445e0828a2c5d5845ccdbd0992fc5eed871d19a",
        "76b2d4d0e09e549ebdc08127a603ba8e81120b94",
        "spatial_memory_evidence.json",
        "spatial_memory_decision.csv",
        "physical_authorization",
        "'authorization': 'none'",
    ]:
        assert required in source

    assert "not a foundation-model or retrieval benchmark" not in source_lower or "foundation_model" in source
    assert "local-simulation-only" in source_lower
    assert "assert canonical_map_state(trusted_map)==before_rejected_loop" in source
    assert "assert rejected_route_after==rejected_route_before" in source
    assert "assert unrelated_update_report['invalidated'] is False" in source
    assert "assert relevant_update_report['invalidated'] is True" in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_advanced_05_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/05-spatial-memory-scene-graphs-navigation/assets")
    expected = {
        "spatial-memory-loop.svg",
        "map-representation-taxonomy.svg",
        "occupancy-evidence.svg",
        "association-errors.svg",
        "object-memory-lifecycle.svg",
        "scene-graph-provenance.svg",
        "hierarchical-navigation.svg",
        "plan-invalidation.svg",
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


def test_advanced_06_contains_the_declared_adaptation_and_continual_learning_lab():
    course = Path("curriculum/advanced/06-multimodal-adaptation-continual-learning")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "BaseModelContract",
        "ShiftContract",
        "CapabilitySuiteContract",
        "ReplayBufferContract",
        "CandidateArtifact",
        "MetricSpec",
        "GateCheck",
        "PromotionDecision",
        "reporting_only_no_changes",
        "TinyDualEncoder",
        "immutable_baseline_hash",
        "AdaptedDualEncoder",
        "linear_probe",
        "projector",
        "adapter",
        "visual_prompt",
        "lora",
        "partial_ft",
        "full_ft",
        "lora_parameter_count",
        "for rank in (1, 2, 4, 8, 16)",
        "rank_sweep",
        "target_gain_B",
        "legacy_regression_A",
        "image_to_text_accuracy",
        "text_to_image_accuracy",
        "paired_cosine",
        "neighborhood_retention",
        "representation_drift",
        "stability_plasticity_controls",
        "pretext_vs_downstream",
        "shortcut_only_full_ft",
        "failure_injection",
        "FROZEN_POLICY_HASH",
        "policy_hash_before_site_c",
        "policy_hash_after_site_c",
        "reporting only; no method, rank, prompt, threshold, buffer, epoch, seed, or rule changes",
        "continual_run",
        "sequential_full_finetune",
        "replay_50",
        "diagonal_fisher",
        "ewc",
        "distillation",
        "forgetting_matrix",
        "signed_improvement",
        "metric_direction_assertions",
        "continual_metric_evidence",
        "reference_semantics",
        "backward_transfer_A",
        "forward_transfer_C_proxy",
        "for capacity in (0, 10, 50, 200)",
        "REPLAY_BUFFER_CONTRACT",
        "source_class_balanced_replay",
        "replay_selection_audit",
        "isolated_adapters",
        "route_adapter",
        "ambiguous_domain",
        "unknown_domain_abstention",
        "wrong_base_blocked",
        "base_checkpoint_digest_mismatch",
        "promote_to_shadow",
        "evaluate_promotion_gate",
        "A_all_required_metrics_pass",
        "B_legacy_metric_fails",
        "C_alignment_metric_missing",
        "D_rollback_artifact_missing",
        "E_suite_version_mismatch",
        'Literal["PASS", "FAIL", "MISSING"]',
        "rollback_target_hash",
        "OPTIONAL_TOOL_MANIFESTS",
        "CV_ENABLE_PEFT = False",
        "CV_ENABLE_AVALANCHE = False",
        "50a277e7c87db460ef7444055788f9da29f2da71",
        "eb075be393e1f458b2c352514ff6c17b5a2c0f4e",
        "multimodal_adaptation_evidence.json",
        "multimodal_adaptation_decision.csv",
        '"foundation_model": False',
        '"authorization": "none"',
    ]:
        assert required in source

    assert "not a foundation model, vlm benchmark, or production adaptation result" in source_lower
    assert "site c is reporting-only" in source_lower
    assert 'make_site("C", "continual_training_experience"' not in source
    assert "assert policy_hash_before_site_c == policy_hash_after_site_c" in source
    assert "assert wrong_base_blocked" in source
    assert "assert promotion_decision.authorization == \"none\"" in source
    assert '"C_alignment_metric_missing": "needs_review"' in source
    assert '"D_rollback_artifact_missing": "reject"' in source
    assert '"E_suite_version_mismatch": "reject"' in source
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")

    readme = (course / "README.md").read_text(encoding="utf-8").lower()
    for required in [
        "improvement on the new task is not sufficient evidence",
        "low rank not imply low forgetting",
        "site c is then reporting-only",
        "training job produces a candidate",
        "replay buffer contract",
        "metric direction is a hard contract",
        "absence of regression evidence is not evidence of no regression",
    ]:
        assert required in readme


def test_advanced_06_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/06-multimodal-adaptation-continual-learning/assets")
    expected = {
        "adaptation-lifecycle.svg",
        "shift-taxonomy.svg",
        "adaptation-methods.svg",
        "lora-low-rank.svg",
        "multimodal-alignment-drift.svg",
        "stability-plasticity.svg",
        "continual-learning-strategies.svg",
        "capability-promotion-gate.svg",
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


def test_advanced_07_contains_the_declared_robustness_uncertainty_and_recovery_lab():
    course = Path("curriculum/advanced/07-robustness-uncertainty-failure-recovery")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "SourceContract",
        "MetricSpec",
        "ScoreSpec",
        "SensorObservation",
        "ReliabilityPolicy",
        "RiskDecision",
        "RecoveryCandidate",
        "RecoveryProposal",
        "VerificationReceipt",
        "TinyReliabilityCNN",
        "ensemble_outputs",
        "expected_calibration_error",
        "brier_score",
        "negative_log_likelihood",
        "Six corruption families × five severities",
        "assert len(corruption_results) == 30",
        "false_confidence_example",
        "mc_dropout_scores",
        "probabilities_at_temperature",
        '"selected_on": "Site B development only"',
        "policy_hash_before_site_c",
        "policy_hash_after_site_c",
        "centroid_distance",
        "mahalanobis",
        "energy",
        "normalize_ood_score",
        "fpr_at_target_tpr",
        '"positive_class": "OOD"',
        "ALL_REJECT_OOD_THRESHOLD",
        "all_reject_operating_point",
        "uncertainty_failure_slices",
        "error_detection_auroc",
        "conformal_quantile",
        "conformal_report",
        "risk_coverage_curve",
        "selective_point",
        "accepted_count",
        "known_answer_selective_risk",
        "MINIMUM_REQUIRED_COVERAGE",
        'risk_coverage["expected_cost_proxy"]',
        "FROZEN_POLICY_HASH",
        'Literal["PASS", "FAIL", "MISSING"]',
        "application_risk_gate",
        "bounded_recovery_policy",
        "alternate_candidate_proposal",
        "ALTERNATE_MINIMUM_SUPPORT",
        "finalize_with_independent_verification",
        "run_bounded_recovery",
        "synthetic_evaluation_oracle",
        "independent_recovery_verification_passed",
        "independent_recovery_verification_failed",
        "recovery_invariant_tests",
        "json_records",
        "allow_nan=False",
        "always_answer",
        "abstain_only",
        "bounded_recovery",
        "OPTIONAL_TOOL_MANIFESTS",
        "robustness_uncertainty_recovery_evidence.json",
        "robustness_uncertainty_recovery_decisions.csv",
        '"authorization": "none"',
    ]:
        assert required in source

    assert "site c is reporting-only" in source_lower
    assert "not a foundation-model benchmark, production reliability result, physical safety case, or deployment authorization" in source_lower
    assert "simulation oracle" in source_lower
    assert source.index("FROZEN_POLICY_HASH") < source.index("site_c_outputs")
    assert "assert policy_hash_before_site_c == policy_hash_after_site_c" in source
    assert "assert (gate_assertions.query(\"case != 'complete evidence'\")[\"result\"] != \"PASS\").all()" in source
    assert "assert (recovery_decisions[\"authorization\"] == \"none\").all()" in source
    assert "assert not ((recovery_decisions[\"terminal_state\"] == \"verified_recovery\") & (~recovery_decisions[\"verified_success\"])).any()" in source
    assert 'assert "verifier" not in bounded_recovery_policy.__code__.co_varnames' in source
    assert 'assert "EVALUATION_ORACLE_LABELS" not in bounded_recovery_policy.__code__.co_names' in source
    assert 'assert all("true_label" not in case for case in cases)' in source
    assert 'known.loc["none accepted", "coverage"] == 0.0 and np.isnan(known.loc["none accepted", "selective_risk"])' in source
    assert 'risk_coverage["selective_risk"].fillna(0)' not in source
    assert 'assert all_reject_operating_point["id_false_reject_rate"] == 1.0' in source
    assert len(notebook["cells"]) == 50
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")

    readme = (course / "README.md").read_text(encoding="utf-8").lower()
    for required in [
        "confidence is not uncertainty",
        "calibration is population-dependent",
        "ood detection is not error detection",
        "exchangeability",
        "risk–coverage",
        "missing calibration, ood, freshness, dependency, or recovery-verification evidence cannot silently become `accept`",
        "attempted recovery is not successful recovery",
        "confidence never grants authority",
    ]:
        assert required in readme


def test_advanced_07_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/07-robustness-uncertainty-failure-recovery/assets")
    expected = {
        "reliability-lifecycle.svg",
        "failure-taxonomy.svg",
        "uncertainty-sources.svg",
        "calibration-reliability.svg",
        "ood-vs-error.svg",
        "risk-coverage.svg",
        "recovery-state-machine.svg",
        "enterprise-reliability-architecture.svg",
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


def test_advanced_08_contains_the_declared_efficient_inference_lab():
    course = Path("curriculum/advanced/08-efficient-spatial-multimodal-inference")
    notebook = json.loads((course / "lab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    source_lower = source.lower()

    for required in [
        "SourceContract",
        "MetricSpec",
        "WorkloadContract",
        "ArtifactContract",
        "GateCheck",
        "DeploymentDecision",
        'Literal["PASS", "FAIL", "MISSING"]',
        "reporting_only_no_changes",
        "TinyDualEncoder",
        "image_to_text_recall",
        "text_to_image_recall_at_5",
        "expected_calibration_error",
        "select_defect_threshold",
        "profile_pipeline",
        "cold_start_ms",
        "steady_p95_ms",
        "resolution_results",
        "false_low_resolution_acceptance_rate",
        "small_evidence_retention_rate",
        "quantize_dequantize_tensor",
        "int8_like_with_output_scale_mismatch",
        "structured_pruning_fraction",
        "task_only_student",
        "multimodal_student",
        "torch.export.export",
        "RUN_OPTIONAL_COMPILE = False",
        "batch_results",
        "simulate_dynamic_batching",
        "saturation_region",
        "bounded_queue_reject_stale",
        "cache_key",
        "STALE_HIT",
        "contract_digest_wrong_tenant",
        "temporal_reuse_results",
        "pareto_mask",
        "candidate_matrix",
        "DEPLOYMENT_BUDGET",
        "POLICY_HASH_BEFORE_SITE_C",
        "POLICY_HASH_AFTER_SITE_C",
        "combined_student_int8_lowres",
        "queue_timeout_rate",
        "efficient_inference_evidence.json",
        "efficient_inference_candidates.csv",
        '"authorization": "none"',
    ]:
        assert required in source

    assert "site c remain reporting-only" in source_lower
    assert "cpu teaching measurements on this notebook host" in source_lower
    assert "not a foundation model or vlm benchmark" in source_lower
    assert "assert unsafe_stale_hit is true" in source_lower
    assert "assert safe_v2_hit is false" in source_lower
    assert "assert policy_hash_before_site_c == policy_hash_after_site_c" in source_lower
    assert "assert decision.authorization == \"none\"" in source_lower
    assert len(notebook["cells"]) == 57
    assert not (course / "lab.py").exists()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")

    readme = (course / "README.md").read_text(encoding="utf-8").lower()
    for required in [
        "performance optimization is a model change",
        "flops, macs, and sparsity are proxies—not runtime",
        "quantization is a capability change",
        "task-only teacher agreement insufficient",
        "a cache hit not necessarily a valid hit",
        "pareto-efficient candidate still be deployment-ineligible",
        "site c remain reporting-only",
    ]:
        assert required in readme


def test_advanced_08_diagrams_are_reusable_and_accessible():
    assets = Path("curriculum/advanced/08-efficient-spatial-multimodal-inference/assets")
    expected = {
        "optimization-lifecycle.svg",
        "pipeline-profile.svg",
        "budget-cascade.svg",
        "token-evidence-retention.svg",
        "compression-lineage.svg",
        "queueing-backpressure.svg",
        "cache-validity.svg",
        "pareto-constraint-gate.svg",
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

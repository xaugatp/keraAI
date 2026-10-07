"""Model 3 (leaf disease identification) pipeline pieces (ADR 0019).

Same shape as `app.ml.leaf_seg`, extended to 3 channels (leaf + 2 diseases):
everything in this package is torch-free (numpy / OpenCV / Pillow only) so the
fast test run can exercise it exhaustively with synthetic data. Only
`app.ml.leaf_disease_identifier.LeafDiseaseIdentifier` touches torch /
segmentation-models-pytorch, and only inside `load()` / `infer()`.

    preprocess   PIL image            -> float32 (1, 3, S, S) network input (reuses Model 2's)
    postprocess  (3, S, S) probs      -> cleaned full-resolution masks + metrics + diagnosis
    overlay      photo + masks        -> the "result.png" visualisation
"""

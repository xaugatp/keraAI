"""Model 2 (leaf segmentation) pipeline pieces.

Everything in this package is torch-free (numpy / OpenCV / Pillow only) so the
fast test run can exercise it exhaustively with synthetic data. Only
`app.ml.leaf_segmenter.LeafSegmenter` touches torch / segmentation-models-pytorch,
and only inside `load()` / `infer()`.

    preprocess   PIL image            -> float32 (1, 3, S, S) network input
    postprocess  (2, S, S) probs      -> cleaned full-resolution masks + metrics + label
    overlay      photo + masks        -> the "result.png" visualisation
"""

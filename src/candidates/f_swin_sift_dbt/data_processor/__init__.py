"""Data adapters for the Swin SIFT-DBT candidate.

Self-contained (no imports from meta_classifier_renewed or full_experiments_using).
Consumes the shared CWT cache at `outputs/cache/data_store_meta_4err.pkl`
built by `EventLockedCWTPipeline(signal_mode='four_error')`.
"""

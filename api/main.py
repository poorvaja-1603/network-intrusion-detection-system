"""
FastAPI inference service for NIDS

Endpoints:
  POST /predict    — Classify network flows (41 raw features)
  GET  /health     — Health check + model version
  GET  /alerts     — Latest attack predictions
  GET  /stats      — Counts per label, total flows, alert rate
"""


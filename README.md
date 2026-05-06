# AI Heartbeat Anomaly Detection — ZX-1

ECG heartbeat classification using Retrieval-Augmented Generation and Large Language Models.

**Student:** Matthew McNair
**Course:** AI7993 — AI Capstone Spring 2026
**Website:** https://mcmcnair2019-sys.github.io/AI-Heartbeat-Anomaly-Detection
**Video:** https://youtu.be/4-KwnCRYee4

## About
This project builds an end-to-end pipeline that converts raw ECG signals from the
MIT-BIH Arrhythmia Dataset into LLM-friendly representations and classifies each
heartbeat as Normal or PVC using a RAG-enhanced reasoning system.

## Dataset
Download the MIT-BIH Arrhythmia Database from:
https://physionet.org/content/mitdb/1.0.0/

## Run
pip install wfdb rank_bm25 sentence-transformers transformers accelerate
python heartbeat_anomaly_fixed.py

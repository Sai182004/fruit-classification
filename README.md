# 🌾 FarmChainX — AI Driven Agricultural Traceability Network

FarmChainX is an integrated digital agriculture platform designed to improve **produce traceability**, **food quality transparency**, and **farm-to-consumer trust** using **AI + data-driven insights**.  
The system enables farmers, aggregators, and consumers to track every stage of the supply chain while providing automated fruit-quality classification through machine learning.

FarmChainX ensures that every product carries a verifiable digital trail—from cultivation to consumption—enhancing credibility and reducing fraud in agricultural markets.

---

## 🚀 Key Features

- **AI-Powered Fruit Classifier (Fresh/Rotten)**  
- **Farmer Registration & Produce Upload Module**
- **Produce Journey Tracking & Ledger**
- **Traceability ID Generation**
- **Supply Chain Record Updates**
- **Buyer/Consumer Traceability Verification**
- **Dashboard for analytics (React + Vite + Tailwind)**
- **Chatbot Assistance (NLP-based)**

---

# 🧪 My Contribution (Python Development – AI Classifier)

I developed the **Python-based Fruit Classification System** using:

- **TensorFlow / Keras**  
- **A custom-trained CNN model (`fruit_cnn.h5`, 112MB)**  
- **Flask REST API** for inference  
- **Top-K predictions, confidence scoring, and feedback saving**  
- **Banana-demo mode for stable presentation**
- **Image preprocessing pipeline + response formatter**

I also created:

- `/api/predict` endpoint  
- `/api/feedback` endpoint to collect correction labels  
- Web UI integration for uploading and classifying images  
- HuggingFace Space deployment using Docker

This classifier is embedded into the FarmChainX dashboard for real-time fruit quality assessment.

---


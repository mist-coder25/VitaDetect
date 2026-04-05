# VitaDetect – Vitamin Deficiency Classification

A clean Flask web application for detecting vitamin deficiencies (A, B, C, D, E) from skin/tissue images using deep learning (Inception v3 transfer learning).

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Project Structure

```
vitamin_final/
├── app.py                  # Main Flask application
├── label_image.py          # TensorFlow inference module
├── retrained_graph.pb      # Trained Inception v3 model
├── retrained_labels.txt    # Class labels
├── requirements.txt
├── templates/
│   ├── home.html           # Landing page
│   ├── register.html       # Registration (first step)
│   ├── login.html          # Login page
│   ├── dashboard.html      # Main prediction interface
│   └── analysis.html       # Dataset analysis charts
└── static/
    ├── css/style.css       # Custom styles
    ├── uploads/            # Temp image uploads
    └── vendor/             # Bootstrap & icons
```

## Features

- ✅ Register → Login → Dashboard flow
- ✅ Secure password hashing (SHA-256)
- ✅ SQLite user database
- ✅ Image upload with drag-and-drop
- ✅ Live AI prediction with result cards
- ✅ Symptoms and dietary recommendations
- ✅ Analysis page with Google Charts
- ✅ Fully responsive design
- ✅ No ocean/restaurant template content

## Important Notes

- Place `retrained_graph.pb` in the root directory (it is large, ~87MB)
- The model was trained using TensorFlow's Inception v3 architecture
- For production, use a proper secret key in `app.secret_key`

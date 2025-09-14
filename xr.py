from flask import Flask, jsonify, request, render_template_string
from werkzeug.utils import secure_filename
import io, os, json, csv, datetime

import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

app = Flask(__name__)

# ---------- Logging ----------
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs.csv")

def append_log(filename: str, pred: str, conf: float, size_bytes: int, ua: str, ip: str):
    newfile = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["ts_iso", "client_ip", "user_agent", "filename", "prediction", "confidence", "size_bytes"])
        w.writerow([
            datetime.datetime.utcnow().isoformat(),
            ip, ua, filename, pred, f"{conf:.6f}", size_bytes
        ])

# ---------- Carga de modelo y preparación ----------
with open("model_meta.json", "r", encoding="utf-8") as f:
    META = json.load(f)
class_names = META["class_names"]
input_size = int(META.get("input_size", 224))

model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load("model_resnet18.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- UI ----------
INDEX_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>X-Ray Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}
    .drop{border:2px dashed #888;border-radius:12px;padding:30px;text-align:center;margin:16px 0}
    .drop.dragover{border-color:#1e88e5;background:#f0f7ff}
    #res{background:transparent;color:#222;padding:0;margin:8px 0 0 0;display:none}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    button{padding:10px 14px;border:0;border-radius:10px;cursor:pointer}
    .ghost{background:#eee}
    .primary{background:#1e88e5;color:white}
  </style>
</head>
<body>
  <h1>Subir imagen de rayos X</h1>

  <div class="row">
    <input id="file" type="file" accept="image/*" style="display:none" />
    <button class="ghost" id="btnPick">Elegir archivo…</button>
    <button class="primary" id="btnSend">Enviar a /predict</button>
  </div>

  <div id="dz" class="drop">
    Arrastra y suelta una imagen aquí
  </div>

  <div id="preview"></div>

  <h2>Resultado</h2>
  <div id="res"></div>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const dz = document.getElementById('dz');
  const fileInput = document.getElementById('file');
  const btnPick = document.getElementById('btnPick');
  const btnSend = document.getElementById('btnSend');
  const preview = document.getElementById('preview');
  const resBox = document.getElementById('res');
  let selectedFile = null;

  function showPreview(file){
    if(!file) return;
    const url = URL.createObjectURL(file);
    preview.innerHTML = '<img src="'+url+'" style="max-width:100%;border-radius:10px;margin:8px 0" />';
  }

  // Botón “Elegir archivo…”
  btnPick.addEventListener('click', () => fileInput.click());

  // Cambio de input file
  fileInput.addEventListener('change', (e) => {
    selectedFile = e.target.files && e.target.files[0] ? e.target.files[0] : null;
    showPreview(selectedFile);
  });

  // Drag & Drop — prevenir navegación y styling
  ['dragenter','dragover','dragleave','drop'].forEach(evt => {
    dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
  });
  dz.addEventListener('dragenter', () => dz.classList.add('dragover'));
  dz.addEventListener('dragover',  () => dz.classList.add('dragover'));
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));

  // Soltar archivo (usar DataTransfer para sincronizar con input)
  dz.addEventListener('drop', (e) => {
    dz.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if(files && files[0]){
      selectedFile = files[0];
      const dt = new DataTransfer();
      dt.items.add(selectedFile);
      fileInput.files = dt.files;
      showPreview(selectedFile);
    }
  });

  // Enviar a /predict
  btnSend.addEventListener('click', async () => {
    if(!selectedFile){ alert('Selecciona o arrastra una imagen.'); return; }
    const fd = new FormData();
    fd.append('file', selectedFile);
    resBox.style.display = 'block';
    resBox.textContent = 'Enviando…';
    try{
      const r = await fetch('/predict', { method:'POST', body: fd });
      const j = await r.json();
      if (j.prediction){
        const isNormal = (j.prediction || "").toUpperCase() === "NORMAL";
        const badgeStyle = `
          display:inline-block;
          padding:4px 8px;
          border-radius:8px;
          font-size:0.95rem;
          ${isNormal ? "background:#1e8e3e;color:#fff" : "background:#d93025;color:#fff"}
        `;
        resBox.innerHTML = `
          <div style="margin-bottom:6px">
            <span style="${badgeStyle}">Predicción: ${j.prediction}</span>
          </div>
          <div style="color:#555;margin:0">Confianza: ${(j.confidence*100).toFixed(1)} %</div>
          <div style="font-size:.9rem;color:#777;margin-top:4px">Archivo: ${j.filename}</div>
        `;
      }else{
        resBox.textContent = JSON.stringify(j, null, 2);
      }
    }catch(err){
      resBox.textContent = 'Error: ' + err;
    }
  });
});
</script>
</body>
</html>
"""

@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

# ---------- /predict ----------
@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify(error="No se envió el archivo con key 'file'"), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify(error="Archivo vacío"), 400

    filename = secure_filename(f.filename)
    data = f.read()

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return jsonify(error=f"No es una imagen válida: {e}"), 400

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred_idx = torch.max(probs, dim=0)

    pred_label = class_names[pred_idx.item()]

    # Registrar en logs
    user_agent = request.headers.get("User-Agent", "")
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    append_log(filename, pred_label, float(conf.item()), len(data), user_agent, client_ip)

    return jsonify({
        "ok": True,
        "filename": filename,
        "prediction": pred_label,
        "confidence": float(conf.item())
    }), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

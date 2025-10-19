# =====================================================
# xr.py – API Flask con modelo Ensamble (descarga automática desde Drive)
# =====================================================

from flask import Flask, jsonify, request, render_template_string
from werkzeug.utils import secure_filename
import io, os, json, csv, datetime, requests
import torch
import torch.nn as nn
from torchvision import models, transforms
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

# ---------- Carga del modelo (Ensambles) ----------
with open("model_meta.json", "r", encoding="utf-8") as f:
    META = json.load(f)

class_names = META["class_names"]
input_size = int(META.get("input_size", 224))

# 🔽 Descarga automática del modelo desde Google Drive si no existe
MODEL_URL = "https://drive.google.com/uc?id=10aeUQAL--ECMQAe_GdBXH1Jm5D45HGt8"
MODEL_PATH = "ensemble_model.pth"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    open(MODEL_PATH, "wb").write(r.content)
    print("✅ Modelo descargado correctamente.")

# ---------- Definición del modelo ----------
class EnsembleResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.models = nn.ModuleList()
        for _ in range(3):  # estructura idéntica al ensamble original
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            self.models.append(m)
    def forward(self, x):
        probs = []
        for m in self.models:
            logits = m(x)
            p = torch.softmax(logits, dim=1)
            probs.append(p)
        return torch.stack(probs).mean(dim=0)

# Cargar pesos del ensamble
model = EnsembleResNet18(num_classes=len(class_names))
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=True)
model.eval()

# ---------- Transformaciones ----------
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Interfaz HTML ----------
INDEX_HTML = """<!doctype html><html lang="es"><head>
<meta charset="utf-8" /><title>Detector de Neumonía – Ensamble</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}
.drop{border:2px dashed #888;border-radius:12px;padding:30px;text-align:center;margin:16px 0}
.drop.dragover{border-color:#1e88e5;background:#f0f7ff}
#res{background:transparent;color:#222;padding:0;margin:8px 0 0 0;display:none}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
button{padding:10px 14px;border:0;border-radius:10px;cursor:pointer}
.ghost{background:#eee}.primary{background:#1e88e5;color:white}
</style></head><body>
<h1>Detector de Neumonía – Ensamble</h1>
<div class="row">
  <input id="file" type="file" accept="image/*" style="display:none" />
  <button class="ghost" id="btnPick">Subir archivo…</button>
  <button class="primary" id="btnSend">Click para predecir</button>
</div>
<div class="drop" id="dz">Arrastra y suelta una imagen aquí</div>
<div id="preview"></div><h2>Resultado</h2><div id="res"></div>
<script>
document.addEventListener('DOMContentLoaded',()=>{const dz=document.getElementById('dz'),fI=document.getElementById('file'),bP=document.getElementById('btnPick'),bS=document.getElementById('btnSend'),pv=document.getElementById('preview'),r=document.getElementById('res');let sf=null;function show(f){if(!f)return;const u=URL.createObjectURL(f);pv.innerHTML='<img src="'+u+'" style="max-width:75%;display:block;margin:12px auto;border-radius:10px" />'}bP.onclick=()=>fI.click();fI.onchange=e=>{sf=e.target.files[0];show(sf)};['dragenter','dragover','dragleave','drop'].forEach(v=>dz.addEventListener(v,e=>{e.preventDefault();e.stopPropagation()},!1));dz.ondrop=e=>{const fs=e.dataTransfer.files;if(fs&&fs[0]){sf=fs[0];const dt=new DataTransfer();dt.items.add(sf);fI.files=dt.files;show(sf)}};bS.onclick=async()=>{if(!sf){alert('Selecciona una imagen.');return;}const fd=new FormData();fd.append('file',sf);r.style.display='block';r.textContent='Procesando…';try{const rr=await fetch('/predict',{method:'POST',body:fd});const j=await rr.json();if(j.prediction){const n=(j.prediction||"").toUpperCase()==="NORMAL";const bs=`display:inline-block;padding:4px 8px;border-radius:8px;font-size:0.95rem;${n?"background:#1e8e3e;color:#fff":"background:#d93025;color:#fff"}`;r.innerHTML=`<div><span style="${bs}">Predicción: ${j.prediction}</span></div><div>Confianza: ${(j.confidence*100).toFixed(1)} %</div><div>Archivo: ${j.filename}</div>`}else r.textContent=JSON.stringify(j,null,2);}catch(e){r.textContent='Error: '+e}}});
</script></body></html>"""

# ---------- Rutas ----------
@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify(error="No se envió archivo"), 400
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
        probs = torch.softmax(outputs[0], dim=0)
        conf, pred_idx = torch.max(probs, dim=0)
    pred_label = class_names[pred_idx.item()]
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
    app.run(host="0.0.0.0", port=5000)

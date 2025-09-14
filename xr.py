from flask import Flask, jsonify, request, render_template_string
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# 1) Health
@app.get("/health")
def health():
    return jsonify(status="ok"), 200

# 2) Home con UI (drag & drop + file input). Hace POST a /predict con fetch.
INDEX_HTML = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Proyecto 7 – API REST (Imagen)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}
    h1{font-size:1.4rem}
    .drop{border:2px dashed #888;border-radius:12px;padding:30px;text-align:center;margin:16px 0}
    .drop.dragover{border-color:#1e88e5;background:#f0f7ff}
    #res{white-space:pre-wrap;background:#111;color:#fafafa;padding:12px;border-radius:10px;display:none}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    button{padding:10px 14px;border:0;border-radius:10px;cursor:pointer}
    .ghost{background:#eee}
    .primary{background:#1e88e5;color:white}
  </style>
</head>
<body>
  <h1>Proyecto 7 – Subir imagen y predecir</h1>

<div class="row">
    <input id="file" type="file" accept="image/*" style="display:none" />
    <button class="ghost" id="btnPick">Elegir archivo…</button>
    <button class="primary" id="btnSend">Enviar a /predict</button>
</div>

  <div id="dz" class="drop">
    Arrastra y suelta una imagen aquí
    <div style="font-size:.9rem;color:#666;margin-top:6px">o usa el botón “Elegir archivo…”</div>
  </div>

  <div id="preview"></div>

  <h2>Respuesta</h2>
  <pre id="res"></pre>

<script>
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

btnPick.onclick = ()=> fileInput.click();

fileInput.onchange = (e)=>{
  selectedFile = e.target.files[0] || null;
  showPreview(selectedFile);
};

['dragenter','dragover'].forEach(evt=>{
  dz.addEventListener(evt, e=>{
    e.preventDefault(); e.stopPropagation();
    dz.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(evt=>{
  dz.addEventListener(evt, e=>{
    e.preventDefault(); e.stopPropagation();
    if(evt==='dragleave') dz.classList.remove('dragover');
  });
});
dz.addEventListener('drop', e=>{
  dz.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if(files && files[0]){
    selectedFile = files[0];
    fileInput.files = files;
    showPreview(selectedFile);
  }
});

btnSend.onclick = async ()=>{
  if(!selectedFile){ alert('Primero selecciona/arrastra una imagen.'); return; }
  const fd = new FormData();
  fd.append('file', selectedFile);
  resBox.style.display='block';
  resBox.textContent='Enviando…';
  try{
    const r = await fetch('/predict', { method:'POST', body: fd });
    const j = await r.json();
    resBox.textContent = JSON.stringify(j, null, 2);
  }catch(err){
    resBox.textContent = 'Error: '+err;
  }
};
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

# 3) /predict — procesa la imagen y responde JSON
@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify(error="No se envió el archivo con key 'file'"), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify(error="Archivo vacío"), 400

    filename = secure_filename(f.filename)
    data = f.read()

    # --------- LÓGICA DE MODELO (placeholder) ----------
    # Aquí luego cargas tu modelo y haces la predicción real.
    # Ejemplo dummy: tamaño del archivo y primera firma de bytes.
    size_bytes = len(data)
    head = list(data[:8])
    # ---------------------------------------------------

    return jsonify({
        "ok": True,
        "filename": filename,
        "size_bytes": size_bytes,
        "bytes_head": head,
        "prediction": "demo-placeholder"  # <-- será reemplazado por tu modelo real
    }), 200

if __name__ == "__main__":
    # Para desarrollo local
    app.run(host="127.0.0.1", port=5000, debug=True)
"""
interfaz_gradio.py
------------------
Interfaz de uso final del analizador de reseñas de hoteles, con tema oscuro.

Ofrece tres pestañas:
  - Reseña individual: estrellas estimadas y sentimiento por aspecto.
  - Analisis por lote: resumen y grafico a partir de un archivo CSV.
  - Metricas del modelo: estadisticas del corpus y comparativa clasico vs BETO.

La interfaz recibe los objetos ya entrenados como parametros, en vez de
depender de variables globales del notebook. Asi el modulo funciona igual
al importarse desde otro entorno.

Uso tipico:

    from interfaz_gradio import crear_interfaz

    app = crear_interfaz(model_beto, tokenizer_beto, prep, analizador, metricas, stats)
    app.launch(share=True)
"""

import re
import traceback

import torch
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr


# Colores base del tema oscuro, reutilizados en graficos y tarjetas
COLOR_FONDO = "#0f1117"
COLOR_PANEL = "#161a23"
COLOR_SENTIMIENTO = {"positivo": "#22c55e", "negativo": "#ef4444", "neutral": "#94a3b8"}
ICONOS = {"positivo": "🟢", "negativo": "🔴", "neutral": "⚪"}

CSS = """
.gradio-container { background:#0f1117 !important; font-family:'Segoe UI',system-ui,sans-serif; color:#e6e9ef !important; }
.card { background:#1c2130; border:1px solid #2a3140; border-radius:16px; padding:22px; margin-top:8px; box-shadow:0 4px 14px rgba(0,0,0,0.35); }
.badge-positivo { background:rgba(34,197,94,.15) !important; color:#4ade80 !important; padding:4px 12px !important; border-radius:12px !important; font-weight:600 !important; border:1px solid #22c55e !important; }
.badge-negativo { background:rgba(239,68,68,.15) !important; color:#f87171 !important; padding:4px 12px !important; border-radius:12px !important; font-weight:600 !important; border:1px solid #ef4444 !important; }
.badge-neutral { background:rgba(148,163,184,.15) !important; color:#cbd5e1 !important; padding:4px 12px !important; border-radius:12px !important; font-weight:600 !important; border:1px solid #64748b !important; }
.aspecto-fila { display:flex; justify-content:space-between; padding:10px 14px; border-bottom:1px solid #2a3140; }
.aspecto-nombre { text-transform:capitalize; color:#e6e9ef; }
.resultado-container { background:#1c2130; border:1px solid #2a3140; border-radius:16px; padding:22px; }
.estrellas-display { font-size:34px; color:#fbbf24; letter-spacing:3px; text-align:center; }
.mensaje-error { color:#f87171; font-weight:bold; }
.stats-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
.stat-box { background:#1c2130; border:1px solid #2a3140; border-radius:14px; padding:16px; text-align:center; }
.stat-num { font-size:22px; font-weight:700; color:#2dd4bf; }
.stat-lbl { font-size:12px; color:#94a3b8; margin-top:4px; }
"""

# Fuerza el tema oscuro de Gradio al abrir la interfaz
JS_DARK = """
function() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def _predecir_estrellas(texto, model_beto, tokenizer_beto, prep):
    """Estima estrellas y sentimiento general a partir de las probabilidades de BETO."""
    texto_limpio = prep.pipeline_transformer(texto)
    device = next(model_beto.parameters()).device
    inputs = tokenizer_beto(texto_limpio, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_beto.eval()
    with torch.no_grad():
        logits = model_beto(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    estrellas = probs[0] * 1.5 + probs[1] * 3 + probs[2] * 4.5
    estrellas = round(estrellas * 2) / 2
    etiqueta = {0: "negativo", 1: "neutral", 2: "positivo"}[int(torch.tensor(probs).argmax())]
    return estrellas, etiqueta


def _tabla_metricas_html(metricas):
    """Construye la tabla comparativa de metricas para la pestaña de resultados."""
    filas = ""
    for nombre, d in metricas.items():
        estilo = "color:#2dd4bf; font-weight:700;" if "BETO" in nombre else "color:#cbd5e1;"
        filas += f"""<tr>
            <td style="padding:10px 14px; {estilo}">{nombre}</td>
            <td style="padding:10px 14px; text-align:center;">{d['Accuracy']:.3f}</td>
            <td style="padding:10px 14px; text-align:center;">{d['Precision']:.3f}</td>
            <td style="padding:10px 14px; text-align:center;">{d['Recall']:.3f}</td>
            <td style="padding:10px 14px; text-align:center;">{d['F1']:.3f}</td></tr>"""
    return f"""
    <div class="card">
      <h3 style="margin-top:0;">Comparativa: clasico vs. moderno</h3>
      <table style="width:100%; border-collapse:collapse; color:#e6e9ef;">
        <tr style="border-bottom:1px solid #334155; color:#94a3b8; font-size:13px;">
          <th style="text-align:left; padding:8px 14px;">Modelo</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        {filas}
      </table>
    </div>"""


def _stats_html(stats):
    """Construye las tarjetas de estadisticas del corpus."""
    tarjetas = ""
    for k, v in stats.items():
        tarjetas += f"""<div class="stat-box"><div class="stat-num">{v}</div><div class="stat-lbl">{k}</div></div>"""
    return f"<div class='stats-grid'>{tarjetas}</div>"


def _grafico_metricas(metricas):
    """Grafico de barras con las metricas globales y el F1 por clase."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(COLOR_PANEL)
    modelos = list(metricas.keys())
    colores = ["#64748b", "#2dd4bf"]

    mets = ["Accuracy", "Precision", "Recall", "F1"]
    w = 0.35
    for i, m in enumerate(modelos):
        vals = [metricas[m][k] for k in mets]
        ax1.bar([p + i * w for p in range(len(mets))], vals, w, label=m, color=colores[i])
    ax1.set_facecolor(COLOR_PANEL)
    ax1.set_ylim(0, 1)
    ax1.set_title("Metricas globales", color="#e6e9ef", fontweight="bold")
    ax1.set_xticks([p + w / 2 for p in range(len(mets))])
    ax1.set_xticklabels(mets, color="#cbd5e1")
    ax1.legend(facecolor=COLOR_PANEL, labelcolor="#e6e9ef", fontsize=8)

    clases = ["negativo", "neutral", "positivo"]
    for i, m in enumerate(modelos):
        vals = [metricas[m]["f1_clase"][c] for c in clases]
        ax2.bar([p + i * w for p in range(len(clases))], vals, w, label=m, color=colores[i])
    ax2.set_facecolor(COLOR_PANEL)
    ax2.set_ylim(0, 1)
    ax2.set_title("F1 por clase", color="#e6e9ef", fontweight="bold")
    ax2.set_xticks([p + w / 2 for p in range(len(clases))])
    ax2.set_xticklabels([c.capitalize() for c in clases], color="#cbd5e1")
    for ax in (ax1, ax2):
        ax.tick_params(colors="#cbd5e1")
    fig.tight_layout()
    return fig


def crear_interfaz(model_beto, tokenizer_beto, prep, analizador, metricas, stats):
    """
    Construye la interfaz Gradio recibiendo los objetos ya entrenados.

    Parametros
    ----------
    model_beto, tokenizer_beto : modelo BETO afinado y su tokenizer.
    prep : instancia de Preprocesador.
    analizador : instancia de AnalizadorAspectos.
    metricas : dict con las metricas de cada modelo (incluye clave 'f1_clase').
    stats : dict con las estadisticas del corpus a mostrar.
    """

    def render_resultado(texto):
        if not texto or not texto.strip():
            return "<p class='mensaje-error'>Por favor pega una reseña.</p>"
        try:
            estrellas, etiqueta = _predecir_estrellas(texto, model_beto, tokenizer_beto, prep)
            aspectos_res = analizador.analizar(texto)
            n = int(round(estrellas))
            estrellas_html = "★" * n + "☆" * (5 - n)
            filas = ""
            for asp, sent in aspectos_res.items():
                filas += f"""<div class="aspecto-fila"><span class="aspecto-nombre">{asp}</span>
                    <span class="badge-{sent}">{ICONOS.get(sent, '⚪')} {sent.capitalize()}</span></div>"""
            return f"""<div class="resultado-container">
                <div style="text-align:center; margin-bottom:16px;">
                    <div class="estrellas-display">{estrellas_html}</div>
                    <div style="color:#94a3b8; font-size:13px; margin-top:2px;">Estrellas estimadas: {estrellas:.1f} / 5</div>
                    <span class="badge-{etiqueta}" style="margin-top:10px; display:inline-block;">Sentimiento general: {etiqueta.capitalize()}</span>
                </div>
                <h4 style="color:#e6e9ef; margin:0 0 10px 4px;">Sentimiento por aspecto</h4>
                <div style="border:1px solid #2a3140; border-radius:10px; overflow:hidden;">{filas}</div>
            </div>"""
        except Exception as e:
            return f"<pre class='mensaje-error'>{e}\n\n{traceback.format_exc()}</pre>"

    def analizar_csv(archivo):
        if archivo is None:
            return "<p class='mensaje-error'>Por favor sube un archivo CSV.</p>", None
        try:
            df = pd.read_csv(archivo.name)
            if "review_text" not in df.columns:
                return "<p class='mensaje-error'>El CSV debe tener una columna 'review_text'.</p>", None
            resultados = df["review_text"].astype(str).apply(
                lambda t: _predecir_estrellas(t, model_beto, tokenizer_beto, prep)[1]
            )
            conteo = resultados.value_counts()
            total = len(resultados)
            filas = ""
            for e in ["positivo", "neutral", "negativo"]:
                pct = (conteo.get(e, 0) / total * 100) if total > 0 else 0
                filas += f"<p style='color:{COLOR_SENTIMIENTO.get(e)}; margin:4px 0;'><b>{e.capitalize()}</b>: {pct:.1f}%</p>"
            resumen = f"<div class='card'><p style='font-size:16px;'>Total analizadas: <b>{total}</b></p>{filas}</div>"

            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor(COLOR_FONDO)
            ax.pie(conteo.values, labels=[k.capitalize() for k in conteo.index], autopct="%1.1f%%",
                   colors=[COLOR_SENTIMIENTO.get(i, "#cccccc") for i in conteo.index],
                   startangle=90, textprops={"fontsize": 12, "color": "#e6e9ef"})
            ax.set_title("Distribucion de sentimientos", color="#e6e9ef", fontweight="bold", pad=18)
            return resumen, fig
        except Exception as e:
            return f"<pre class='mensaje-error'>{e}\n\n{traceback.format_exc()}</pre>", None

    with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), js=JS_DARK) as demo:
        gr.Markdown("<h1 style='text-align:center; color:#f1f5f9; margin-bottom:0;'>Analizador de Reseñas de Hoteles</h1>")
        gr.Markdown("<p style='text-align:center; color:#94a3b8; margin-top:5px;'>Sentimiento por aspecto, estrellas estimadas y comparativa clasico vs. Transformer</p>")

        with gr.Tab("Reseña individual"):
            entrada = gr.Textbox(lines=6, label="Reseña",
                                 placeholder="Ej: El hotel estaba limpio y el personal amable, aunque el desayuno dejo que desear...")
            boton = gr.Button("Analizar", variant="primary")
            salida = gr.HTML()
            boton.click(fn=render_resultado, inputs=entrada, outputs=salida)
            gr.Examples(examples=[
                ["El hotel estaba impecable, el personal super amable, pero el desayuno dejo mucho que desear."],
                ["Pesima atencion en recepcion, la habitacion sucia y muy caro para lo que ofrece."],
                ["Todo excelente, la mejor experiencia de nuestras vacaciones, sin duda regresariamos."],
            ], inputs=entrada)

        with gr.Tab("Analisis por lote (CSV)"):
            archivo = gr.File(label="Sube un CSV con columna 'review_text'", file_types=[".csv"])
            boton_csv = gr.Button("Analizar lote", variant="primary")
            resumen_html = gr.HTML()
            grafico = gr.Plot()
            boton_csv.click(fn=analizar_csv, inputs=archivo, outputs=[resumen_html, grafico])

        with gr.Tab("Metricas del modelo"):
            gr.Markdown("<h3 style='color:#f1f5f9;'>Estadisticas del corpus</h3>")
            gr.HTML(_stats_html(stats))
            gr.HTML(_tabla_metricas_html(metricas))
            gr.Plot(value=_grafico_metricas(metricas))

    return demo

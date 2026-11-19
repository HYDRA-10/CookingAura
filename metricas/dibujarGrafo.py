"""Dibuja los nodos y aristas del grafo de CookingAura.

Uso (desde la raíz del backend, o dentro del contenedor):
    python -m src.agents.dibujar_grafo
    python -m src.agents.dibujar_grafo --formatos mermaid html
    python -m src.agents.dibujar_grafo --salida /app/data/grafo

Genera en la carpeta de salida (por defecto data/grafo, montada en el host):
    grafo.mmd   -> código Mermaid (pégalo en https://mermaid.live o en un README de GitHub)
    grafo.html  -> página que dibuja el grafo al abrirla en el navegador
    grafo.png   -> imagen (usa el servicio mermaid.ink: necesita internet)
y muestra el grafo en ASCII por consola si está instalado 'grandalf'.

Aristas continuas = fijas · aristas discontinuas = condicionales.
"""
import argparse
import html
from pathlib import Path
# 1. Calculamos la ruta absoluta hacia la carpeta "backend" y la añadimos al path
import sys
import os
ruta_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(ruta_backend)
from src.agents.graph import app

FORMATOS = ("mermaid", "html", "png", "ascii")
SALIDA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "data" / "grafo"

PLANTILLA_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grafo de CookingAura</title>
<style>
  :root {{ --fondo: #ffffff; --texto: #1f2328; --suave: #59636e; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --fondo: #0d1117; --texto: #e6edf3; --suave: #9198a1; }} }}
  body {{ margin: 0; padding: 2rem; background: var(--fondo); color: var(--texto);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
  p {{ color: var(--suave); margin: 0 0 1.5rem; }}
  .lienzo {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>👨‍🍳 Grafo de CookingAura</h1>
<p>{nodos} nodos · {aristas} aristas — continuas: fijas · discontinuas: condicionales</p>
<div class="lienzo"><pre class="mermaid">{mermaid}</pre></div>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
  const oscuro = window.matchMedia("(prefers-color-scheme: dark)").matches;
  mermaid.initialize({{ startOnLoad: true, theme: oscuro ? "dark" : "default" }});
</script>
</body>
</html>
"""


def resumen(grafo) -> str:
    """Listado de nodos y aristas en texto plano (no necesita nada extra)."""
    lineas = [f"Nodos ({len(grafo.nodes)}):"]
    lineas += [f"  · {nombre}" for nombre in grafo.nodes]
    lineas.append(f"\nAristas ({len(grafo.edges)}):")
    for arista in grafo.edges:
        tipo = "condicional" if arista.conditional else "fija"
        lineas.append(f"  {arista.source:>22} → {arista.target:<22} ({tipo})")
    return "\n".join(lineas)


def dibujar(app, carpeta: Path, formatos: tuple[str, ...] = FORMATOS) -> None:
    """Exporta el grafo compilado 'app' en los formatos pedidos."""
    grafo = app.get_graph()
    carpeta.mkdir(parents=True, exist_ok=True)
    mermaid = grafo.draw_mermaid()

    print(resumen(grafo), "\n")

    if "mermaid" in formatos:
        ruta = carpeta / "grafo.mmd"
        ruta.write_text(mermaid, encoding="utf-8")
        print(f"✅ Mermaid: {ruta}")

    if "html" in formatos:
        ruta = carpeta / "grafo.html"
        # Se escapa porque las etiquetas de LangGraph llevan HTML (<p>...</p>);
        # Mermaid lee el texto ya desescapado del <pre>.
        ruta.write_text(PLANTILLA_HTML.format(
            mermaid=html.escape(mermaid), nodos=len(grafo.nodes), aristas=len(grafo.edges)
        ), encoding="utf-8")
        print(f"✅ HTML:    {ruta}  (ábrelo en el navegador)")

    if "png" in formatos:
        ruta = carpeta / "grafo.png"
        try:
            ruta.write_bytes(grafo.draw_mermaid_png())   # renderiza vía https://mermaid.ink
            print(f"✅ PNG:     {ruta}")
        except Exception as e:
            print(f"⚠️  PNG no generado ({e}). Usa grafo.html o pega grafo.mmd en https://mermaid.live")

    if "ascii" in formatos:
        try:
            print("\n" + grafo.draw_ascii())
        except ImportError:
            print("ℹ️  Para ver el grafo en ASCII: pip install grandalf")
        except Exception as e:
            print(f"⚠️  ASCII no disponible: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dibuja el grafo LangGraph de CookingAura.")
    parser.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO,
                        help=f"Carpeta de salida (por defecto: {SALIDA_POR_DEFECTO})")
    parser.add_argument("--formatos", nargs="+", choices=FORMATOS, default=list(FORMATOS),
                        help="Formatos a generar (por defecto, todos)")
    args = parser.parse_args()

    
    dibujar(app, args.salida, tuple(args.formatos))


if __name__ == "__main__":
    main()
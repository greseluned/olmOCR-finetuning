#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
limpiar_salida.py

Extrae el texto transcrito (natural_text) de la salida JSON nativa de olmOCR,
tolerando JSON truncado o mal formado, para calcular CER/WER solo sobre el
texto. extraer_natural_text(salida) devuelve (texto, formato), donde formato es
"json", "recuperado" o "sin_json" (con sufijo "+esc" si se decodificaron
escapes literales). 05_inference.py lo importa; también sirve como utilidad
para limpiar TXT ya generados con la salida cruda.

Uso:
    python limpiar_salida.py ruta/directorio_con_txt/

    Reescribe cada TXT con solo el texto extraído y guarda el original en
    <directorio>/raw/ (sin sobrescribir copias existentes). Los TXT que ya son
    texto plano no se tocan.
"""

import json
import re
import sys
from pathlib import Path

FORMATO_JSON       = "json"
FORMATO_RECUPERADO = "recuperado"
FORMATO_SIN_JSON   = "sin_json"

_RE_CLAVE     = re.compile(r'"natural_text"\s*:\s*')
_RE_FENCE_INI = re.compile(r"^```[a-zA-Z]*\s*")
_RE_FENCE_FIN = re.compile(r"\s*```$")
_RE_UNICODE_INCOMPLETO = re.compile(r"\\u[0-9a-fA-F]{0,3}$")
_RE_ESCAPE = re.compile(r'\\(["\\/nrt])')
_MAPA_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "r": "\r", "t": "\t"}
SUFIJO_ESC = "+esc"


def _decodificar_cadena(resto: str) -> str:
    i, n = 0, len(resto)
    while i < n:
        c = resto[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            break
        i += 1
    seg = resto[:min(i, n)]

    # Si la salida se cortó a mitad de un escape, se descarta el escape roto.
    barras = len(seg) - len(seg.rstrip("\\"))
    if barras % 2 == 1:
        seg = seg[:-1]
    seg = _RE_UNICODE_INCOMPLETO.sub("", seg)

    try:
        return json.loads('"' + seg + '"', strict=False)
    except json.JSONDecodeError:
        return (seg.replace("\\n", "\n").replace("\\t", "\t")
                   .replace('\\"', '"').replace("\\\\", "\\"))


def _extraer_bruto(salida: str) -> tuple[str, str]:
    s = salida.strip()
    if s.startswith("```"):
        s = _RE_FENCE_FIN.sub("", _RE_FENCE_INI.sub("", s)).strip()

    # 1) JSON completo
    try:
        d = json.loads(s, strict=False)
        if isinstance(d, dict) and "natural_text" in d:
            nt = d["natural_text"]
            return ("" if nt is None else str(nt)).strip(), FORMATO_JSON
    except json.JSONDecodeError:
        pass

    # 2) JSON truncado o mal formado
    m = _RE_CLAVE.search(s)
    if m:
        resto = s[m.end():]
        if resto.startswith('"'):
            return _decodificar_cadena(resto[1:]).strip(), FORMATO_RECUPERADO
        return "", FORMATO_RECUPERADO

    # 3) Sin el esquema de olmOCR: texto plano
    return salida.strip(), FORMATO_SIN_JSON


def _desescapar_si_literal(texto: str) -> tuple[str, bool]:
    if "\\n" in texto and "\n" not in texto:
        decodificado = _RE_ESCAPE.sub(lambda m: _MAPA_ESCAPES[m.group(1)], texto)
        return decodificado.strip(), True
    return texto, False


def extraer_natural_text(salida: str) -> tuple[str, str]:
    texto, formato = _extraer_bruto(salida)
    texto, cambiado = _desescapar_si_literal(texto)
    if cambiado:
        formato += SUFIJO_ESC
    return texto, formato


def limpiar_directorio(dir_txt: Path) -> None:
    dir_raw = dir_txt / "raw"
    n_mod = n_plano = 0
    for ruta in sorted(dir_txt.glob("*.txt")):
        raw = ruta.read_text(encoding="utf-8")
        texto, formato = extraer_natural_text(raw)
        if formato == FORMATO_SIN_JSON:
            n_plano += 1
            continue
        dir_raw.mkdir(exist_ok=True)
        ruta_raw = dir_raw / ruta.name
        if not ruta_raw.exists():
            ruta_raw.write_text(raw, encoding="utf-8")
        ruta.write_text(texto, encoding="utf-8")
        n_mod += 1
    print(f"{n_mod} TXT limpiados (originales en {dir_raw}); "
          f"{n_plano} ya eran texto plano.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        print("Uso: python salida_olmocr.py ruta/directorio_con_txt/")
        sys.exit(1)
    limpiar_directorio(Path(sys.argv[1]))
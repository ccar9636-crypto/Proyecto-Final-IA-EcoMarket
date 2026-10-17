"""
Herramientas (tools) del agente de EcoMarket.
Son funciones simuladas: leen los JSON de docs/ en lugar de una base real.
Cada una recibe datos simples (texto) y devuelve un diccionario con "exito" (True/False).
"""

import json
import os
import uuid
from datetime import date, datetime

from langchain_core.tools import tool

# Fecha "de hoy" para el simulador. Se fija para que la demo sea reproducible
# (los pedidos de prueba son de septiembre 2026). Se puede cambiar con FECHA_REFERENCIA.
FECHA_REFERENCIA = os.getenv("FECHA_REFERENCIA", "2026-09-25")
PLAZO_DIAS = 30  # Política: 30 días desde la recepción


def _cargar(nombre: str):
    with open(f"docs/{nombre}", "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def verificar_elegibilidad_producto(tracking_number: str, producto_id: str) -> dict:
    """Verifica si un producto de un pedido puede devolverse según la política de EcoMarket.
    Parámetros: tracking_number (ej. 'ECO-1001') y producto_id (ej. 'PROD-002').
    Devuelve: {'exito', 'elegible', 'motivo'}."""
    tracking_number, producto_id = tracking_number.strip().upper(), producto_id.strip().upper()

    pedido = next((p for p in _cargar("mock_orders.json") if p["tracking_number"] == tracking_number), None)
    if not pedido:
        return {"exito": False, "elegible": False, "motivo": f"El pedido {tracking_number} no existe."}

    producto = next((p for p in _cargar("inventario_productos.json") if p["id"] == producto_id), None)
    if not producto:
        return {"exito": False, "elegible": False, "motivo": f"El producto {producto_id} no existe."}

    if pedido.get("producto_id") != producto_id:
        return {"exito": True, "elegible": False, "motivo": "Ese producto no pertenece al pedido indicado."}
    if pedido["estado"] != "Entregado":
        return {"exito": True, "elegible": False,
                "motivo": f"El pedido está '{pedido['estado']}'; solo se devuelven pedidos entregados."}
    if not producto["admite_devolucion"]:
        return {"exito": True, "elegible": False,
                "motivo": f"'{producto['nombre']}' ({producto['categoria']}) no admite devolución por normativa "
                          "de salud. Alternativa: bono de compra o nota de crédito."}

    dias = (datetime.strptime(FECHA_REFERENCIA, "%Y-%m-%d").date()
            - datetime.strptime(pedido["fecha_entrega"], "%Y-%m-%d").date()).days
    if dias > PLAZO_DIAS:
        return {"exito": True, "elegible": False,
                "motivo": f"Han pasado {dias} días desde la entrega; el plazo máximo es {PLAZO_DIAS}."}

    return {"exito": True, "elegible": True,
            "motivo": f"Elegible: entregado hace {dias} días (límite {PLAZO_DIAS})."}


@tool
def generar_etiqueta_devolucion(tracking_number: str, producto_id: str, motivo: str) -> dict:
    """Genera la etiqueta de devolución. Úsala SOLO después de confirmar que el producto es elegible.
    Parámetros: tracking_number, producto_id y motivo ('defectuoso', 'error_ecomarket' o 'arrepentimiento').
    Devuelve: {'exito', 'etiqueta', 'costo_envio_usd', 'mensaje'}."""
    # Seguridad: la herramienta vuelve a validar, así el agente no puede saltarse la política.
    check = verificar_elegibilidad_producto.invoke({"tracking_number": tracking_number, "producto_id": producto_id})
    if not check["elegible"]:
        return {"exito": False, "mensaje": f"No se generó etiqueta. {check['motivo']}"}

    gratis = motivo.strip().lower() in ("defectuoso", "error_ecomarket")
    etiqueta = f"DEV-{tracking_number.upper()}-{uuid.uuid4().hex[:6].upper()}"
    _registrar(etiqueta, tracking_number, producto_id, motivo)
    return {"exito": True, "etiqueta": etiqueta,
            "costo_envio_usd": 0 if gratis else 5,
            "mensaje": "Etiqueta generada. Reembolso en 5 a 8 días hábiles tras inspección."}


def _registrar(etiqueta, tracking, producto, motivo):
    """Auditoría: guarda cada etiqueta emitida en un archivo de registro."""
    with open("registro_acciones.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | ETIQUETA {etiqueta} | {tracking} | {producto} | {motivo}\n")


HERRAMIENTAS_ACCION = [verificar_elegibilidad_producto, generar_etiqueta_devolucion]

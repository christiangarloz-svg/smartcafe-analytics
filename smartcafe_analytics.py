"""
SmartCafé Analytics - Versión Pro Mejorada
Sistema inteligente de análisis y monitoreo para cafeterías.
"""

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------- Configuración Estética (HUD Pro) ------------------------------ #
COLOR_QUEUE = (255, 140, 0)      # Azul/Naranja moderno (BGR) -> Zona de espera
COLOR_COUNTER = (0, 230, 118)    # Verde esmeralda moderno -> Mostrador / Barista
COLOR_ALERT = (0, 0, 255)        # Rojo alerta
COLOR_TEXT = (245, 245, 245)     # Blanco suave
COLOR_PANEL_BG = (20, 20, 20)    # Fondo oscuro elegante
GRACE_SECONDS = 2.0              # Tiempo de gracia en mostrador


@dataclass
class Customer:
    track_id: int
    first_seen: float
    queue_start: float | None = None
    counter_start: float | None = None
    last_in_counter: float | None = None
    finished: bool = False


# ------------------------------ Utilidades -------------------------------- #
def point_in_roi(pt, roi):
    x, y = pt
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def default_rois(w, h):
    queue = (int(0.05 * w), int(0.35 * h), int(0.52 * w), int(0.95 * h))
    counter = (int(0.55 * w), int(0.35 * h), int(0.95 * w), int(0.95 * h))
    return {"queue": queue, "counter": counter}


def select_rois(frame):
    print("Selecciona la ZONA DE ESPERA y presiona ENTER...")
    qx, qy, qw, qh = cv2.selectROI("Zona de espera", frame, showCrosshair=False)
    cv2.destroyWindow("Zona de espera")
    print("Selecciona la ZONA DE MOSTRADOR y presiona ENTER...")
    cx, cy, cw, ch = cv2.selectROI("Zona de mostrador", frame, showCrosshair=False)
    cv2.destroyWindow("Zona de mostrador")
    return {
        "queue": (qx, qy, qx + qw, qy + qh),
        "counter": (cx, cy, cx + cw, cy + ch),
    }


def fmt(seconds):
    if seconds is None:
        return "--"
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def draw_panel(frame, lines, alert=False):
    """Panel flotante con estilo minimalista moderno y esquinas semitransparentes."""
    pad, line_h = 12, 26
    w = 350
    h = pad * 2 + line_h * len(lines)
    
    overlay = frame.copy()
    # Dibujar rectángulo de fondo sólido oscuro
    cv2.rectangle(overlay, (15, 15), (15 + w, 15 + h), COLOR_PANEL_BG, -1)
    # Aplicar transparencia (efecto vidrio esmerilado)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Borde sutil del panel
    border_color = COLOR_ALERT if alert else (80, 80, 80)
    cv2.rectangle(frame, (15, 15), (15 + w, 15 + h), border_color, 2)

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (15 + pad, 15 + pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)
        
    if alert:
        # Alerta parpadeante en todo el marco de la ventana
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), COLOR_ALERT, 4)


# ------------------------------- Principal -------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="SmartCafé Analytics Pro")
    ap.add_argument("--source", default="0", help="0 para webcam o ruta a un video")
    ap.add_argument("--model", default="yolov8n.pt", help="modelo YOLO")
    ap.add_argument("--conf", type=float, default=0.45, help="confianza mínima")
    ap.add_argument("--max-wait", type=float, default=30.0, help="límite de espera en segundos")
    ap.add_argument("--select-roi", action="store_true", help="dibujar ROIs")
    ap.add_argument("--log", default="registro_atencion_pro.csv", help="CSV de salida")
    args = ap.parse_args()

    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No se pudo leer el primer cuadro.")
    h, w = frame.shape[:2]

    rois = select_rois(frame) if args.select_roi else default_rois(w, h)
    model = YOLO(args.model)

    customers: dict[int, Customer] = {}
    wait_times: list[float] = []
    service_times: list[float] = []
    log_rows = []

    frame_idx = 0
    t_prev = time.time()
    fps_show = 0.0
    if not is_webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[INFO] Sistema SmartCafé Analytics Pro iniciado. Presiona 'q' para salir, 'r' para reiniciar.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time() if is_webcam else frame_idx / fps
        frame_idx += 1

        res = model.track(frame, persist=True, classes=[0], conf=args.conf,
                          tracker="bytetrack.yaml", verbose=False)[0]

        in_queue_ids, in_counter_ids = set(), set()
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            ids = res.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                #foot = ((x1 + x2) // 2, y2)
                foot = ((x1 + x2) // 2, y1 + int((y2 - y1) * 0.6))
                c = customers.setdefault(tid, Customer(tid, first_seen=now))

                in_q = point_in_roi(foot, rois["queue"])
                in_c = point_in_roi(foot, rois["counter"])

                if in_q:
                    in_queue_ids.add(tid)
                    if c.queue_start is None and c.counter_start is None:
                        c.queue_start = now

                if in_c:
                    in_counter_ids.add(tid)
                    if c.counter_start is None:
                        c.counter_start = now
                        if c.queue_start is not None:
                            wait_times.append(now - c.queue_start)
                    c.last_in_counter = now

                # Dibujo elegante de las cajitas de los clientes
                color = COLOR_COUNTER if in_c else COLOR_QUEUE if in_q else (180, 180, 180)
                
                # Rectángulo con esquinas estéticas simuladas (grosores limpios)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Etiqueta flotante superior
                label = f"ID {tid}"
                if in_q and c.queue_start is not None and c.counter_start is None:
                    label += f" | esp: {fmt(now - c.queue_start)}"
                elif in_c and c.counter_start is not None:
                    label += f" | aten: {fmt(now - c.counter_start)}"
                
                # Fondo para el texto de la etiqueta para mayor legibilidad
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.circle(frame, foot, 5, color, -1)

        # Cierre de atenciones
        for c in customers.values():
            if (c.counter_start is not None and not c.finished
                    and c.last_in_counter is not None
                    and now - c.last_in_counter > GRACE_SECONDS):
                duration = c.last_in_counter - c.counter_start
                c.finished = True
                if duration >= 1.0:
                    service_times.append(duration)
                    wait_dur = (c.counter_start - c.queue_start) if c.queue_start else 0.0
                    log_rows.append([
                        datetime.now().isoformat(timespec="seconds"), 
                        c.track_id,
                        round(duration, 1),
                        round(wait_dur, 1),
                        round(duration + wait_dur, 1) # Tiempo total de estancia
                    ])

        # Métricas
        waiting_now = [c for c in customers.values()
                       if c.track_id in in_queue_ids and c.counter_start is None
                       and c.queue_start is not None]
        current_waits = [now - c.queue_start for c in waiting_now]
        max_current_wait = max(current_waits) if current_waits else 0.0
        alert = max_current_wait > args.max_wait

        all_waits = wait_times + current_waits
        avg_wait = float(np.mean(all_waits)) if all_waits else None
        avg_service = float(np.mean(service_times)) if service_times else None

        # Dibujar Zonas ROI con estilo profesional
        for name, color in (("queue", COLOR_ALERT if alert else COLOR_QUEUE), ("counter", COLOR_COUNTER)):
            x1, y1, x2, y2 = rois[name]
            # Rectángulo semitransparente sutil dentro de la zona ROI
            roi_overlay = frame.copy()
            cv2.rectangle(roi_overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(roi_overlay, 0.08, frame, 0.92, 0, frame)
            
            # Marco de la ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            title = "ZONA DE ESPERA (COLA)" if name == "queue" else "MOSTRADOR / BARISTA"
            cv2.putText(frame, title, (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2, cv2.LINE_AA)

        # FPS
        t_now = time.time()
        fps_show = 0.9 * fps_show + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        # Panel de métricas actualizado
        lines = [
            ("☕ SmartCafe Analytics Pro", (0, 220, 255)),
            (f"Clientes en cuadro: {len(in_queue_ids | in_counter_ids)}", COLOR_TEXT),
            (f"En espera: {len(waiting_now)} | En barra: {len(in_counter_ids)}", COLOR_TEXT),
            (f"Clientes atendidos: {len(service_times)}", COLOR_TEXT),
            (f"Espera promedio: {fmt(avg_wait)}", COLOR_TEXT),
            (f"Atención promedio: {fmt(avg_service)}", COLOR_TEXT),
            (f"Límite de espera: {fmt(args.max_wait)}", (200, 200, 200)),
            (f"FPS de procesamiento: {fps_show:.1f}", (150, 150, 150)),
        ]
        if alert:
            lines.append((f"¡ALERTA!: Espera > límite ({fmt(max_current_wait)})", COLOR_ALERT))
        
        draw_panel(frame, lines, alert)

        cv2.imshow("SmartCafe Analytics Pro", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            customers.clear(); wait_times.clear(); service_times.clear(); log_rows.clear()

    cap.release()
    cv2.destroyAllWindows()

    # Guardar registro CSV optimizado
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["timestamp", "id_cliente", "tiempo_atencion_s", "tiempo_espera_s", "tiempo_total_estancia_s"])
        wr.writerows(log_rows)
    print(f"[INFO] Reporte profesional guardado en {args.log} ({len(log_rows)} registros).")


if __name__ == "__main__":
    main()
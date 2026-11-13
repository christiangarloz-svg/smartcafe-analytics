# SmartCafé Analytics Pro ☕🤖

Sistema inteligente de análisis y monitoreo en tiempo real para cafeterías y negocios gastronómicos, desarrollado con **Python, OpenCV y YOLOv8 (ByteTrack)** como parte de los proyectos prácticos de Ingeniería de Software con IA.

---

## 📌 Requerimientos del Sistema y Módulos Funcionales
* **Detección y Conteo de Clientes:** Procesamiento en tiempo real (webcam o video) utilizando modelos de visión artificial para identificar personas en la zona de pedidos o mesas[cite: 7].
* **Cronometraje de Tiempos de Servicio (Barista / Atención):** Mecanismo visual mediante Regiones de Interés (ROI) configurable para registrar automáticamente el tiempo de espera en cola y el tiempo de atención en el mostrador[cite: 7].
* **Interfaz Gráfica / Dashboard Informativo (HUD):** Superposición de métricas en tiempo real que incluye número de clientes actuales, espera promedio estimada y alertas visuales automáticas si se supera el tiempo límite establecido[cite: 7].

---

## ⚙️ Instrucciones de Instalación y Dependencias

Asegúrate de tener Python instalado en tu entorno y ejecuta el siguiente comando en la terminal para instalar las dependencias necesarias:

```bash
pip install ultralytics opencv-python numpy
🚀 Guía de Uso y Calibración
Puedes ejecutar el sistema utilizando tu cámara web (por defecto) o un archivo de video pregrabado:

Usando la Webcam (por defecto):

Bash
python smartcafe_analytics.py --source 0
Usando un archivo de video:

Bash
python smartcafe_analytics.py --source barra.mp4
Calibración Interactiva de Regiones de Interés (ROI):
Si deseas dibujar manualmente las zonas de espera y mostrador con el mouse sobre el cuadro de video, añade el parámetro --select-roi:

Bash
python smartcafe_analytics.py --source barra.mp4 --select-roi
Teclas de Control durante la ejecución:

Presiona q para salir de la aplicación y exportar el reporte consolidado en formato CSV (registro_atencion_pro.csv).

Presiona r para reiniciar las métricas y contadores en caliente.

Desarrollado para la gestión eficiente de flujos operativos y analítica de datos en establecimientos comerciales.

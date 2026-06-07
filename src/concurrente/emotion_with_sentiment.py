"""
ES: Script para parchear colecciones de MongoDB existentes añadiendo el análisis de
    emotion_with_sentiment a los documentos que no lo tienen o tienen valores "NA"/nulos.
EN: Script to patch existing MongoDB collections by adding emotion_with_sentiment
    analysis to documents that are missing it or have "NA"/null values.
"""

import json
import time as timer
import concurrent.futures
import threading
from pymongo import MongoClient
import sentiment_analysis.llm_call as llm_call
from dataset_preprocessing.emotion_mapper import normalize_emotion_label
import utils


# ── Configuración ────────────────────────────────────────────────────────────

MONGO_URI = 'mongodb://localhost:27017'       # ES: Cambiar si la BD está en otro equipo
                                              # EN: Change if the DB is on another machine
DATABASE_NAME = 'TFG_Results_EmotionalAnalysis'
NUM_THREADS = 8


# ── Helpers ──────────────────────────────────────────────────────────────────

def needs_patch(doc, llm_chosen):
    """
    ES: Devuelve True si el documento necesita ser parcheado (campo ausente, None o "NA").
    EN: Returns True if the document needs patching (field absent, None, or "NA").
    """
    field = f"emotion_raw_with_sentiment_{llm_chosen}"
    value = doc.get(field)
    return value is None or value == "NA" or str(value).strip() == ""


def list_patchable_collections(db, llm_chosen):
    """
    ES: Lista las colecciones de la BD que corresponden al LLM elegido.
    EN: Lists collections in the DB that correspond to the chosen LLM.
    """
    suffix = f"_{llm_chosen}_results_concurrente"
    return sorted([name for name in db.list_collection_names() if name.endswith(suffix)])


# ── Lógica de parcheo ────────────────────────────────────────────────────────

def patch_document(doc, llm_chosen, justification, collection, dataset, stats, stats_lock):
    """
    ES: Procesa un único documento: llama al LLM con el sentiment ya guardado y
        actualiza el documento en MongoDB con los campos emotion_with_sentiment.
    EN: Processes a single document: calls the LLM with the already-saved sentiment
        and updates the document in MongoDB with the emotion_with_sentiment fields.
    """
    doc_id   = doc["_id"]
    text     = doc.get("text", "")
    msg_id   = doc.get("id", str(doc_id))
    sentiment = doc.get(f"sentiment", "")

    if not sentiment or sentiment == "NA":
        with stats_lock:
            stats["skipped"] += 1
        print(f"  [SKIP] doc {msg_id} — sentiment vacío, no se puede parchear.")
        return

    try:
        func = llm_call.LLM_FUNCTIONS[llm_chosen.upper()]
        (emotion_with_sentiment_raw,
         emotion_with_sentiment_mapped,
         certainty_emotion_with_sentiment,
         justification_emotion_with_sentiment,
         date_emotion_with_sentiment,
         time_emotion_with_sentiment) = func(
            text, justification, "emotion_with_sentiment_analysis", dataset, msg_id,
            sentiment=sentiment
        )

        update_fields = {
            f"emotion_raw_with_sentiment_{llm_chosen}":     emotion_with_sentiment_raw,
            f"emotion_mapped_with_sentiment_{llm_chosen}":  emotion_with_sentiment_mapped,
            f"certainty_emotion_with_sentiment_{llm_chosen}": certainty_emotion_with_sentiment,
            f"processing_date_emotion_with_sentiment":      date_emotion_with_sentiment,
            f"processing_hour_emotion_with_sentiment":      time_emotion_with_sentiment,
        }
        if justification:
            update_fields[f"justification_emotion_with_sentiment_{llm_chosen}"] = justification_emotion_with_sentiment

        collection.update_one({"_id": doc_id}, {"$set": update_fields})

        with stats_lock:
            stats["patched"] += 1
        print(f"  [OK]   doc {msg_id} → {emotion_with_sentiment_mapped} ({certainty_emotion_with_sentiment})")

    except Exception as e:
        with stats_lock:
            stats["errors"] += 1
        print(f"  [ERROR] doc {msg_id}: {e}")


def patch_collection(collection, llm_chosen, justification, dataset):
    """
    ES: Itera sobre los documentos de la colección que necesitan parcheo y los procesa
        concurrentemente con ThreadPoolExecutor.
    EN: Iterates over documents in the collection that need patching and processes
        them concurrently with ThreadPoolExecutor.
    """
    docs_to_patch = [doc for doc in collection.find() if needs_patch(doc, llm_chosen)]
    total = len(docs_to_patch)

    if total == 0:
        print("No hay documentos que parchear en esta colección.")
        return

    print(f"Documentos a parchear: {total}")

    stats = {"patched": 0, "skipped": 0, "errors": 0}
    stats_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(patch_document, doc, llm_chosen, justification, collection, dataset, stats, stats_lock)
            for doc in docs_to_patch
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(f"\nResumen: {stats['patched']} parcheados | {stats['skipped']} saltados | {stats['errors']} errores")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]

        # ES: Elegir LLM
        # EN: Choose LLM
        llm_chosen = input(f"¿Qué LLM quieres parchear? ({'/'.join(llm_call.AVAILABLE_LLMS)}): ").strip()
        while llm_chosen.upper() not in llm_call.AVAILABLE_LLMS:
            llm_chosen = input(f"Formato incorrecto. ({'/'.join(llm_call.AVAILABLE_LLMS)}): ").strip()
        llm_chosen = llm_chosen.lower()

        # ES: Listar colecciones disponibles para ese LLM
        # EN: List available collections for that LLM
        available = list_patchable_collections(db, llm_chosen)
        if not available:
            raise ValueError(f"No se encontraron colecciones para el LLM '{llm_chosen}' en la base de datos '{DATABASE_NAME}'.")

        print("\nColecciones disponibles:")
        for i, name in enumerate(available, start=1):
            print(f"  {i}) {name}")

        selected_collection_name = None
        while selected_collection_name is None:
            choice = input("Selecciona la colección por número o nombre exacto: ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(available):
                    selected_collection_name = available[idx - 1]
            else:
                if choice in available:
                    selected_collection_name = choice
            if selected_collection_name is None:
                print("Selección inválida, inténtalo de nuevo.")

        # ES: Extraer el nombre del dataset del nombre de la colección
        # EN: Extract the dataset name from the collection name
        suffix = f"_{llm_chosen}_results_concurrente"
        dataset = selected_collection_name.replace(suffix, "")

        # ES: Justificación
        # EN: Justification
        str_just = input("¿Quieres justificación en el análisis? (Y/N): ").strip().upper()
        while str_just not in ("Y", "N"):
            str_just = input("Formato incorrecto (Y/N): ").strip().upper()
        justification = str_just == "Y"

        collection = db[selected_collection_name]

        print(f"\nIniciando parcheo de '{selected_collection_name}' con {NUM_THREADS} hilos...\n")
        start = timer.perf_counter()
        patch_collection(collection, llm_chosen, justification, dataset)
        elapsed = round(timer.perf_counter() - start, 4)
        print(f"\nTiempo total de parcheo: {elapsed} segundos")

    except Exception as e:
        print(f"Error: {e}")
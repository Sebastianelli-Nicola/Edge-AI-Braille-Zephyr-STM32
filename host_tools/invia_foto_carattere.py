import serial
import time
import os
import random  
from PIL import Image
import numpy as np
from tqdm import tqdm  

# =========================================================
# 1. CONFIGURAZIONE E DIZIONARIO CARATTERI
# =========================================================
PORTA_SERIALE = 'COM6' 
BAUD_RATE = 115200
CARTELLA_DATASET = r'test\X_test_salvato' 
NUMERO_TEST = 1543 
FILE_REPORT = 'report_stress_test_braille.txt'

# Mappatura completa delle 65 classi del modello (0-64)
# Qualsiasi classe non presente o non riconosciuta verrà mappata come "Sfondo"
DIZIONARIO_BRAILLE = {
    0: "Sfondo",
    1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J",
    11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T",
    21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z",
    27: "1", 28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9", 36: "0",
    37: ",", 38: ";", 39: ":", 40: ".", 41: "?", 42: "!", 43: "'", 44: "-", 45: '"', 46: "(",
    47: ")", 48: "*", 49: "/", 50: "\\", 51: "@", 52: "#", 53: "$", 54: "%", 55: "&", 56: "+",
    57: "=", 58: "<", 59: ">", 60: "[", 61: "]", 62: "{", 63: "}", 64: "_"
}

DIZIONARIO_BRAILLE_CORRETTO = {
    0: "Sfondo",
    1: "A", 2: ",", 3: "'", 4: "", 5: "", 6: "MAIUSCOLO", 7: "B", 8: "K", 9: "C", 10: "E",
    11: "", 12: ";", 13: "I", 14: ":", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T",
    21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z",
    27: "1", 28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9", 36: "0",
    37: ",", 38: ";", 39: ":", 40: ".", 41: "?", 42: "!", 43: "'", 44: "-", 45: '"', 46: "(",
    47: ")", 48: "*", 49: "/", 50: "\\", 51: "@", 52: "#", 53: "$", 54: "%", 55: "&", 56: "+",
    57: "=", 58: "<", 59: ">", 60: "[", 61: "]", 62: "{", 63: "}", 64: "_"
}

print(f"Connessione alla STM32 sulla porta {PORTA_SERIALE}...")
try:
    ser = serial.Serial(PORTA_SERIALE, BAUD_RATE, timeout=2.0)
    ser.reset_input_buffer() 
except Exception:
    print("Errore: Porta seriale non trovata.")
    exit()

percorsi_immagini = []
for root, dirs, files in os.walk(CARTELLA_DATASET):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            percorsi_immagini.append(os.path.join(root, file))

if len(percorsi_immagini) > 0:
    random.shuffle(percorsi_immagini) 
    percorsi_immagini = percorsi_immagini[:NUMERO_TEST] 

totale_immagini = len(percorsi_immagini)
if totale_immagini == 0:
    print("Nessuna immagine trovata!")
    exit()

print(f"Trovate {totale_immagini} immagini! Inizio dello Stress Test Rapido...")

corrette = 0
analizzate = 0
tempi_inferenza = [] 
lista_errori = []    
registro_completo_test = [] # Memorizza l'esito di ogni singola immagine per il documento finale

# =========================================================
# 2. CICLO DI TEST CON BARRA DI PROGRESSO (TQDM)
# =========================================================
barra = tqdm(percorsi_immagini, desc="Valutazione", unit="img", leave=True)

for percorso in barra:
    nome_file = os.path.basename(percorso)
    
    try:
        id_classe_vera = int(nome_file.upper().split("CLASS_")[1].split(".")[0])
    except Exception:
        continue 
    
    # Traduzione ID in carattere (se non esiste nel dizionario, diventa "Sfondo")
    carattere_vero = DIZIONARIO_BRAILLE.get(id_classe_vera, "Sfondo")
    
    try:
        # --- A. HANDSHAKE ---
        pronto = False
        ser.reset_input_buffer()
        while not pronto:
            ser.write(b'S') 
            inizio_hs = time.time()
            while time.time() - inizio_hs < 0.5:
                if ser.in_waiting > 0:
                    if "READY" in ser.readline().decode('utf-8', errors='ignore'):
                        pronto = True
                        break
                time.sleep(0.01)

        # --- B. PREPARAZIONE IMMAGINE ---
        img = Image.open(percorso).convert('L').resize((32, 32))
        pixel_data = np.array(img, dtype=np.uint8).flatten()

        # Inizio cronometro per la misurazione della Latenza
        start_time = time.time()

        # --- C. INVIO A BLOCCHI ---
        for i in range(32):
            ser.write(pixel_data[i*32 : (i+1)*32])
            time.sleep(0.002) 
        
        # --- D. LETTURA RISPOSTA ---
        id_classe_letta = -1
        mentre_legge = True
        
        while mentre_legge:
            if ser.in_waiting > 0:
                risposta = ser.readline().decode('utf-8', errors='ignore').strip()
                if "Lettera Letta:" in risposta:
                    id_classe_letta = int(risposta.split(":")[-1].strip())
                elif "Sicurezza:" in risposta:
                    mentre_legge = False 
            else:
                time.sleep(0.01)
        
        # Fine cronometro Latenza
        end_time = time.time()
        latenza_ms = (end_time - start_time) * 1000
        tempi_inferenza.append(end_time - start_time)
                    
        # Traduzione ID letto in carattere (se non esiste nel dizionario, diventa "Sfondo")
        carattere_letto = DIZIONARIO_BRAILLE.get(id_classe_letta, "Sfondo")
        
        # --- E. VALUTAZIONE ---
        analizzate += 1
        
        if id_classe_vera == id_classe_letta:
            corrette += 1
            esito = "CORRETTO"
            registro_completo_test.append(f"[{esito}] File: {nome_file} | Reale: '{carattere_vero}' -> Predetto: '{carattere_letto}' | Tempo: {latenza_ms:.1f} ms")
        else:
            esito = "ERRORE"
            stringa_errore = f"File: {nome_file} | Reale: '{carattere_vero}' -> Predetto: '{carattere_letto}'"
            lista_errori.append(stringa_errore)
            registro_completo_test.append(f"[{esito}] {stringa_errore} | Tempo: {latenza_ms:.1f} ms")
            
        # Aggiorna il testo di fianco alla barra di progresso
        accuratezza_attuale = (corrette / analizzate) * 100
        barra.set_postfix({"Acc": f"{accuratezza_attuale:.1f}%"})

    except Exception:
        pass

# =========================================================
# 3. REPORT FINALE A SCHERMO
# =========================================================
accuratezza_totale = (corrette / analizzate) * 100 if analizzate > 0 else 0
tempo_medio_ms = (sum(tempi_inferenza) / len(tempi_inferenza)) * 1000 if tempi_inferenza else 0

print("\n" + "=" * 60)
print(" 📊 STRESS TEST MICRO-SSD COMPLETATO!")
print("=" * 60)
print(f" 🖼️ Immagini Analizzate  : {analizzate}")
print(f" ✅ Caratteri Corretti  : {corrette}")
print(f" ❌ Caratteri Errati    : {analizzate - corrette}")
print(f" ⏱️ Latenza Media Edge   : {tempo_medio_ms:.1f} ms / img")
print(f" 🎯 ACCURATEZZA FINALE   : {accuratezza_totale:.2f}%")
print("=" * 60)

if lista_errori:
    print("\n--- 🔍 ESTRATTO DEGLI ERRORI SUI CARATTERI (Primi 15) ---")
    for errore in lista_errori[:15]:
        print(f"  - {errore}")
    if len(lista_errori) > 15:
        print(f"  ... e altri {len(lista_errori) - 15} errori omessi.")
print("\n")
ser.close()

# =========================================================
# 4. SALVATAGGIO DEI RISULTATI NEL DOCUMENTO DI TEST
# =========================================================
print(f"💾 Salvataggio di tutti i test in corso nel file '{FILE_REPORT}'...")
try:
    with open(FILE_REPORT, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("            REPORT DETTAGLIATO DI STRESS TEST BRAILLE       \n")
        f.write("============================================================\n\n")
        f.write(f"Data Esecuzione        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Immagini Totali Target : {NUMERO_TEST}\n")
        f.write(f"Immagini Effettive     : {analizzate}\n")
        f.write(f"Previsioni Corrette    : {corrette}\n")
        f.write(f"Previsioni Errate      : {analizzate - corrette}\n")
        f.write(f"Latenza Media Computo  : {tempo_medio_ms:.2f} ms\n")
        f.write(f"Accuratezza Finale     : {accuratezza_totale:.2f}%\n")
        f.write("\n" + "-"*60 + "\n")
        f.write("📊 REGISTRO COMPLETO DI TUTTE LE INFERENZE\n")
        f.write("-"*60 + "\n")
        
        for riga in registro_completo_test:
            f.write(riga + "\n")
            
        f.write("\n" + "-"*60 + "\n")
        f.write("❌ ELENCO COMPLETO DEI SOLI ERRORI SUI CARATTERI\n")
        f.write("-"*60 + "\n")
        if lista_errori:
            for errore in lista_errori:
                f.write(f"  • {errore}\n")
        else:
            f.write("  Nessun errore registrato. Accuratezza strutturale al 100%!\n")
            
    print("✅ Salvataggio completato con successo!")
except Exception as e:
    print(f"⚠️ Errore durante il salvataggio del file: {e}")
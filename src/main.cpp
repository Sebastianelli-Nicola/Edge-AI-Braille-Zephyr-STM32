// ==============================================================================
// FIRMWARE: Braille Edge AI su STM32 (Zephyr RTOS + TensorFlow Lite Micro)
// DESCRIZIONE: Sistema ottimizzato per l'inferenza real-time di caratteri Braille
// ==============================================================================

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/uart.h> 
#include <math.h> 

// Librerie di TensorFlow Lite per Microcontrollori
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// File generato dal convertitore che contiene l'array dei pesi del modello AI
#include "model.h"

// --- CONFIGURAZIONE LED E UART ---
// Otteniamo i riferimenti hardware dal DeviceTree di Zephyr
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
const struct device *uart_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_console)); 

// --- TASK DEL LED (Blinky) ---
// Questo task indipendente fa lampeggiare il LED per confermare che l'OS è vivo.
void blinky_entry(void *, void *, void *) {
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
    while (1) {
        gpio_pin_toggle_dt(&led);
        k_msleep(500); // Pausa di mezzo secondo (Non blocca il resto del sistema)
    }
}
// K_THREAD_DEFINE lancia automaticamente il task all'avvio.
// Lo stack è stato ridotto al minimo indispensabile (256 byte) per risparmiare RAM.
K_THREAD_DEFINE(blinky_id, 256, blinky_entry, NULL, NULL, NULL, 4, 0, 0);


// --- CONFIGURAZIONE INTELLIGENZA ARTIFICIALE ---
// Allochiamo l'Area di Memoria (Tensor Arena) in cui l'IA farà i suoi calcoli.
// 80 KB sono calibrati perfettamente per il modello "Leggero" a Convoluzioni Separabili.
constexpr int kTensorArenaSize = 80 * 1024; 
alignas(16) static uint8_t tensor_arena[kTensorArenaSize]; // L'allineamento a 16 byte velocizza gli accessi in memoria

// --- TASK PRINCIPALE DELL'IA ---
void ai_entry(void *, void *, void *) {
    k_msleep(1000); // Breve pausa iniziale per stabilizzare l'hardware all'accensione
    
    printk("\n========================================\n");
    printk(" Avvio Sistema Braille Edge AI (Modello Leggero)\n");
    printk("========================================\n");

    // Verifica che la comunicazione Seriale col PC sia pronta
    if (!device_is_ready(uart_dev)) {
        printk("Errore: Seriale non pronta!\n");
        return; // Ferma tutto se non c'è connessione
    }

    // Carica la struttura del modello dalla Flash (Array C++)
    const tflite::Model* model = tflite::GetModel(g_braille_ssd_model_data);
    
    // Configurazione del Resolver: Dichiariamo SOLO le operazioni matematiche 
    // fisicamente presenti nel nostro modello. Questo risparmia tantissima memoria.
    tflite::MicroMutableOpResolver<16> resolver; 
    resolver.AddFullyConnected(); 
    resolver.AddRelu();           
    resolver.AddSoftmax();        
    resolver.AddReshape();        
    resolver.AddConv2D();         
    resolver.AddDepthwiseConv2D(); // Operazione chiave che rende il modello "Leggero"
    resolver.AddMaxPool2D();      
    resolver.AddLogistic();       
    resolver.AddShape();          
    resolver.AddStridedSlice();   
    resolver.AddPack();
    resolver.AddMul();            
    resolver.AddMean();           
    resolver.AddAdd();             
    resolver.AddSub(); 
    resolver.AddPad();             // Necessario per le Convoluzioni in determinati padding          

    // Inizializza l'interprete unendo Modello, Operazioni e Memoria RAM (Arena)
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    
    // Tenta di allocare la memoria. Se fallisce, l'Arena è troppo piccola o manca un'operazione nel resolver.
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printk("Errore critico: RAM TensorArena insufficiente o Layer Mancante!\n");
        return;
    }
    
    printk("Modello allocato con successo in RAM!\n");

    // Creiamo i puntatori diretti agli Ingressi e alle Uscite per la massima velocità
    TfLiteTensor* input = interpreter.input(0);
    
    // SISTEMA DI SICUREZZA: Evita crash se viene caricato per errore un modello sbagliato
    if (input->type != kTfLiteFloat32 || input->bytes != 1024 * sizeof(float)) {
        printk("ERRORE: Il modello non e' Float32 o non accetta 1024 valori (32x32)!\n");
        return;
    }
    
    // Il nostro modello ha 2 output (Logits per la classe, e Confidenza per la presenza)
    // Troviamo dinamicamente qual è l'uno e qual è l'altro in base alle loro dimensioni.
    TfLiteTensor* out0 = interpreter.output(0);
    TfLiteTensor* out1 = interpreter.output(1);
    TfLiteTensor* output_logits = (out0->bytes > out1->bytes) ? out0 : out1;
    TfLiteTensor* output_conf   = (out0->bytes < out1->bytes) ? out0 : out1;

    // --- LOOP INFINITO DELL'APPLICAZIONE ---
    while (1) {
        unsigned char rx_char;
        
        // --- 0. SINCRONIZZAZIONE (HANDSHAKE) ---
        // La scheda rimane in attesa di ricevere il carattere 'S' (Sync) da Python.
        // Qui k_msleep(1) è SICURO perché Python manda un solo carattere alla volta e poi si ferma.
        while (true) {
            if (uart_poll_in(uart_dev, &rx_char) == 0) {
                if (rx_char == 'S') {
                    break; // Sincronizzazione avvenuta, usciamo dal ciclo di attesa
                }
            } else {
                k_msleep(1); // Mette in pausa il task per 1ms per non usurare la CPU a vuoto
            }
        }

        // Segnala a Python che siamo pronti a ricevere l'immagine
        printk("READY\n");

        // --- 1. FASE DI INPUT (RICEZIONE IMMAGINE - VELOCITÀ MASSIMA) ---
        // Riceviamo 1024 byte ad altissima velocità. Nessun comando "sleep" qui, altrimenti 
        // il buffer hardware andrebbe in Overrun e perderemmo pezzi della foto.
        for (int i = 0; i < 1024; i++) {
            
            // Tight Spinloop: la CPU gira al 100% per afferrare il byte al volo
            while (uart_poll_in(uart_dev, &rx_char) < 0) { }
            
            // Normalizzazione [0, 1]. TRUCCO PRESTAZIONALE:
            // Usiamo la moltiplicazione (* 0.00392...) invece della divisione (/ 255.0f) 
            // perché l'hardware ARM esegue le moltiplicazioni molto più velocemente.
            input->data.f[i] = rx_char * 0.003921568f; 
        }

        // --- 2. FASE DI INFERENZA ---
        // L'IA elabora l'immagine. È il blocco che richiede più tempo
        if (interpreter.Invoke() != kTfLiteOk) {
            printk("Errore durante l'inferenza!\n");
            continue;
        }
            
        // --- 3. FASE DI ESTRAZIONE E STAMPA RISULTATI ---
        
        // Calcolo della probabilità (Sigmoide) sulla confidenza di "presenza lettera".
        // TRUCCO PRESTAZIONALE: usiamo expf() (Singola precisione 32-bit hardware) 
        // e non exp() (Doppia precisione 64-bit software lenta e pesante in RAM).
        float prob = 1.0f / (1.0f + expf(-output_conf->data.f[0]));

        // Ricerca dell'indice con la probabilità più alta (ArgMax) sui 65 logit di output.
        int best_idx = 0;
        float best_prob = output_logits->data.f[0];
        
        for (int i = 1; i < 65; i++) {
            if (output_logits->data.f[i] > best_prob) {
                best_prob = output_logits->data.f[i];
                best_idx = i;
            }
        }

        // Se la probabilità generale (confidenza) supera il 50%, stampa la lettera trovata,
        // altrimenti stampa '0' indicando che il modello ritiene ci sia solo rumore/sfondo.
        if (prob > 0.5f) {
            printk("Lettera Letta: %d\n", best_idx);
        } else {
            printk("Lettera Letta: 0\n"); 
        }
        
        // Moltiplichiamo per 100 e convertiamo in Intero per semplificare la formattazione stringa (%d)
        printk("Sicurezza: %d%%\n", (int)(prob * 100));
    }
}
// Avvia il task principale dell'Intelligenza Artificiale assegnandogli molta priorità (5) e uno stack capiente.
K_THREAD_DEFINE(ai_id, 16384, ai_entry, NULL, NULL, NULL, 5, 0, 0);
# 🧠🔋🚁 Predikce zbytkovýho času letu

## 🤖 Co to je? 👉👉

Systém 👨‍💻 provádí měření 🔋 napětí a dalších veličin, zpracovává 📊 data prostřednictvím neuronové sítě 🧠 a odhaduje, jak dlouho může bezpilotní prostředek 🚁 ještě bezpečně ✨operovat✨

---

## 🧪 Funkcionalita systému 🤖🧠⚡💾📉

- provádí sběr a měření 👾📡📊 telemetrických dat z bezpilotního prostředku 🚁  
- zpracovává 📈📚 a analyzuje 📉 vstupy pomocí neuronových sítí 🧠🔥🧮  
- generuje vizualizace 📊✨  
- exportuje 📤 výsledky a predikce do CSV formátu 📄💾📂 pro další zpracování (nebo ne)  

---

## 📁 Soubory 📂📄🧠🪫📉📊🔧📦📡

| 📄 Soubor           | 🧾 Popis                                                |
|---------------------|---------------------------------------------------------|
| `bat_map_lnu.py`    | predikce 🔋 napětí 🧠 model 🧪📊                         |
| `batmot_map_lnu.py` | predikce 🔋 + výkon 🔧 motorů 🧠📉                       |
| `final_map.py`      | trénink 📈 modelu 🧠 export 📤                           |
| `sigpred_lnu.py`    | predikce 🔮 budoucího 📊 signálu ⏩                     |
| `batpred.py`        | realtime ⏱️📉 inference 🧠📲                             |
| `flight.py`         | řízení 🚁 letu 🎮📡 záznam 📄                            |
| `maketab.py`        | výpočty 🧮 energie ⚡ trajektorie 🧭 grafy 📊             |
| `read.py`           | joystick 🎮 input 🔌 mapping 📍                         |

---

## 🎮💀 JOYSTICK 🧃📡

Funguje to – ale jen s **PS4 DualShockem** 😃😃😃 (ano, tímhle přesně 👉🎮)  

Jiný ovladač?  
❌ ne  

Mohlo by se přepsat `read.py` a upravit mapování vstupů 🎮🧩  
ale... **PROČ TO DĚLAT**  
✨ život je krátkej ✨  
💸 ovladač stojí míň než ten čas, co by to zabralo 💸  

**DualShock nebo nic**  

---

## 🛠️ Spuštění systému 🚀📡🔧📂🧠📊

- Připojit dron pomocí rádiového rozhraní 🔗📶  
- Spustit skript `flight.py` pro aktivaci řízení letu 🧠🛫  
- Pro trénování modelů použít skripty `final_map.py` nebo `bat*_lnu.py` 🧪📈  
- Výstupy (grafy, predikce, CSV soubory) jsou ukládány do složky `bat_pics/` 📁📄📊📤  
- ¯\\\_(ツ)\_/¯  

---

## 🕳️ Stav systému a čitelnosti 📉📂🧠🔁

Projekt je ve fázi, kdy je možné provést **refaktoring** 🔁, nebo ne 📡🧎‍♂️  
Struktura kódu je propojená ↔️ – moduly závisí jeden na druhém 🪢 a některé části se opakují 2× nebo vícekrát ♻️ (nebylo to záměrné)  
Názvy proměnných 📛 odpovídají vývoji systému v reálném čase 🕒, často bez následné revize ✍️  
Udržovatelnost je omezená ⚠️, orientace je možná, ale vyžaduje trpělivost ⏳  

---

## 📦 Závislosti a prostředí 🔧📦💻🧪

- `torch` 🔥 – neuronové sítě, výpočty  
- `pandas` 🐼 – zpracování datových rámců  
- `matplotlib` 🎨 – vizualizace výstupů  
- `scikit-learn` 🎓 – metriky, pomocné funkce  
- `cflib` 📡 – komunikace s platformou Crazyflie  

Instalaci lze provést prostřednictvím `pip` 📦

---

## 💀 Závěr systému ⚰️📉🧠

🪫 Projekt funguje  
📈 Predikuje  
📂 Ukládá  
🔁 A cyklicky opakuje stejné chyby  

📡 Pokud byl kód otevřen, je již pozdě na návrat  
🧠 Pokud běží, neměnit  
📉 Pokud selže, ... zavřít editor

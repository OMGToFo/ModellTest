import streamlit as st
import joblib
import pandas as pd
import io
import pickle

st.title("📊 GRP-Prognose Tool mit Modell-Upload")

# Modell-Upload
uploaded_model_file = st.file_uploader("📁 Lade ein GRP-Vorhersagemodell (.pkl)", type=["pkl"])

model = None
if uploaded_model_file is not None:
    try:
        # Laden des Modells aus dem hochgeladenen File
        #model = joblib.load(uploaded_model_file) oroinal code
        #model = pickle.load(uploaded_model_file)
        #model = joblib.load(io.BytesIO(uploaded_model_file.read()))
        model = joblib.load(io.BytesIO(uploaded_model_file.read()))

        st.success("✅ Modell erfolgreich geladen!")
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {str(e)}")

# Eingabefelder nur anzeigen, wenn ein Modell geladen wurde
if model:
    st.header("🔢 Eingabedaten")

    sportart = st.text_input("Sportart", value="Fussball")
    wochentag = st.selectbox("Wochentag", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    uhrzeit = st.number_input("Uhrzeit (Stunde)", min_value=0, max_value=23, value=20)
    zielgruppe = st.selectbox("Zielgruppe", ["14-29", "14-49", "30-59", "60+"])

    if st.button("🚀 Prognose starten"):
        try:
            df = pd.DataFrame([{
                "Sportart": sportart,
                "Wochentag": wochentag,
                "Uhrzeit": uhrzeit,
                "Zielgruppe": zielgruppe
            }])
            pred = model.predict(df)[0]
            st.success(f"📈 Prognostizierte GRP: {round(pred, 2)}")
        except Exception as e:
            st.error(f"❌ Fehler bei der Prognose: {str(e)}")
else:
    st.info("Bitte lade zuerst ein Modell hoch, um das Tool zu verwenden.")
